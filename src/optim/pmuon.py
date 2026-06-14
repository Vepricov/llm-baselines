"""
PMuon (Preconditioned Muon).

Port of the pmuon optimizer from the modded-nanogpt speedrun record
(zzp1012/modded-nanogpt, branch pmuon-track3-3225,
records/track_3_optimization/results/20260507_pmuon), adapted to the
llm-baselines Muon conventions.

PMuon = Muon with covariance preconditioning: instead of polar(momentum) it
computes polar(A^{-gamma} @ u @ C^{-gamma}), where A (left) and C (right) are
streaming gradient-covariance estimates and A^{-gamma}, C^{-gamma} are matrix
powers computed via a streaming orthogonal basis (one QR power-iteration step) +
in-basis eigenvalue power. The Newton-Schulz polar step then orthogonalizes.

Unlike Shampoo's direct inverse-root, the eigenvalues here are re-estimated in the
streaming basis and the final polar discards magnitudes, so the preconditioner is
robust to imperfect factor eigenvalues -- which makes this a good host for the
DyKAF proj_split factors (see pmuon_dykaf, planned).
"""

import os

import torch
import torch.distributed as dist

from .muon import zeropower_via_newtonschulz5


@torch.no_grad()
def _streaming_cov_power(C, state, key, gamma, eps=1e-6):
    """A^{-gamma} via a streaming orthogonal basis (QR power iteration) + in-basis
    eigenvalue power. Verbatim from the pmuon record."""
    n = C.size(0)
    Q = state.get(key, None)
    if Q is None or Q.shape != (n, n) or Q.device != C.device:
        Q, _ = torch.linalg.qr(
            torch.randn(n, n, device=C.device, dtype=C.dtype), mode="reduced"
        )
    Q, _ = torch.linalg.qr(C @ Q, mode="reduced")
    state[key] = Q.detach()
    lam = (Q * (C @ Q)).sum(dim=0).clamp_min(eps)
    d = lam.pow(-gamma)
    res = Q @ torch.diag(d) @ Q.T
    res *= n**0.5 / (d.norm() + eps)
    return res


class PMuon(torch.optim.Optimizer):
    """
    Preconditioned Muon. Same interface as Muon (see src/optim/muon.py) plus:
        gamma: covariance matrix-power exponent (0.3 default).
        cov_beta: EMA factor for the covariance buffers (0.95 default).
    Parameters that are {0,1}-D or look like embed/head (size(0) >= 10000) fall
    back to AdamW, exactly as in Muon.
    """

    def __init__(
        self,
        muon_params,
        lr=0.035,
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        gamma=0.3,
        cov_beta=0.95,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            gamma=gamma,
            cov_beta=cov_beta,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        for p in muon_params:
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]["use_muon"] = True
            else:
                self.state[p]["use_muon"] = False
        for p in adamw_params:
            self.state[p]["use_muon"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self):
        for group in self.param_groups:
            ############################
            #          PMuon           #
            ############################
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            momentum = group["momentum"]
            gamma = group["gamma"]
            cov_beta = group["cov_beta"]
            eps = 1e-6

            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["cov_buf"] = torch.zeros(
                            g.size(1), g.size(1), device=g.device, dtype=g.dtype
                        )
                        state["out_buf"] = torch.zeros(
                            g.size(0), g.size(0), device=g.device, dtype=g.dtype
                        )
                    buf = state["momentum_buffer"]

                    # --- pmuon momentum (EMA + nesterov), as in the record ---
                    buf.lerp_(g, 1 - momentum)
                    u = g.lerp(buf, momentum) if group["nesterov"] else buf

                    # --- streaming covariance estimates of the update u ---
                    cov_ = u.mT @ u
                    state["cov_buf"].mul_(cov_beta).add_(cov_)
                    C_reg = torch.add(cov_, state["cov_buf"], alpha=cov_beta)
                    C_reg.diagonal().add_(eps)

                    out_ = u @ u.mT
                    state["out_buf"].mul_(cov_beta).add_(out_)
                    A_reg = torch.add(out_, state["out_buf"], alpha=cov_beta)
                    A_reg.diagonal().add_(eps)

                    # --- A^{-gamma} u C^{-gamma}, then polar via Newton-Schulz ---
                    C_neg = _streaming_cov_power(C_reg, state, "cov_Q", gamma)
                    A_neg = _streaming_cov_power(A_reg, state, "out_Q", gamma)
                    upd = zeropower_via_newtonschulz5(
                        A_neg @ u @ C_neg, steps=group["ns_steps"]
                    )
                    upd *= max(1, upd.size(0) / upd.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = upd.flatten()
                curr_idx += p.numel()

            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in params:
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["adamw_lr_ratio"] * group["lr"]
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (adamw_eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)
