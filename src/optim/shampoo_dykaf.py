"""
Distributed Shampoo with the DyKAF preconditioner.

This is the Shampoo optimizer (see ``src/optim/shampoo.py``::``DistributedShampoo``)
with a single change: the left/right preconditioners ``L`` and ``R`` are maintained
by the fast DyKAF projector-splitting routine ``proj_split`` (from
``src/optim/dykaf_with_parallel_proj_split.py``) instead of the full-matrix
exponential moving average of ``g g^T`` / ``g^T g``.

The Shampoo update rule itself (``L^{-1/2} g R^{-1/2}`` blended with the Adam
moment) is left untouched. Sides whose dimension exceeds ``max_precond_dim``
(e.g. the token embedding) are not preconditioned, mirroring DyKAF/SOAP.
"""

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

from optim.dykaf_with_parallel_proj_split import proj_split


def _is_active(M) -> bool:
    """A preconditioner side is active when it is a square 2D matrix."""
    return torch.is_tensor(M) and M.dim() == 2


def _inv_sqrt(M: torch.Tensor, damping: float, eps: float) -> torch.Tensor:
    """
    Matrix inverse square root ``M^{-1/2}`` via eigendecomposition.

    Unlike SOAP/DyKAF, Shampoo inverts L/R, so the (normalized, possibly
    rank-deficient) factors returned by ``proj_split`` must be ridge-damped
    relative to their own scale to bound the amplification of null-space
    directions. We clamp negative eigenvalues (numerical noise / indefiniteness
    in the projector-split factor) to zero and add a relative ridge
    ``damping * lambda_max`` plus an absolute floor before taking ``lambda^{-1/2}``.

    NB: the genuine matrix inverse square root ``Q diag(lambda^{-1/2}) Q^T`` is
    used here, not an element-wise ``sqrt`` of the matrix inverse.
    """
    M = 0.5 * (M + M.transpose(-1, -2))  # symmetrize away numerical asymmetry
    eigvals, eigvecs = torch.linalg.eigh(M)
    eigvals = eigvals.clamp_min(0)
    lam_max = eigvals.max().clamp_min(eps)
    inv_sqrt_eigvals = (eigvals + damping * lam_max + eps).rsqrt()
    return (eigvecs * inv_sqrt_eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)


class DyKAFShampoo(torch.optim.Optimizer):
    """
    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients for the first and
            second moment running averages (default: (0.9, 0.999)).
        eps (float, optional): Term added to denominators for numerical
            stability (default: 1e-8).
        weight_decay (float, optional): Decoupled weight decay (default: 0).
        shampoo_decay (float, optional): Decay (beta) of the L/R preconditioner
            moving average, passed to ``proj_split`` (default: 0.9).
        init (str, optional): Initialization of the L/R factors used by
            ``proj_split`` ("eps", "kron", or "zeros") (default: "eps").
        max_precond_dim (int, optional): Sides larger than this are not
            preconditioned (default: 10000).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        shampoo_decay: float = 0.9,
        init: str = "eps",
        max_precond_dim: int = 10000,
        damping: float = 1e-4,
        precondition_frequency: int = 10,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if len(betas) != 2:
            raise ValueError(f"Invalid betas length: {len(betas)}, expected 2.")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Invalid betas: {betas}. Each beta must be in [0, 1).")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= shampoo_decay < 1.0:
            raise ValueError(
                f"Invalid shampoo_decay value: {shampoo_decay}. Must be in [0, 1)."
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            shampoo_decay=shampoo_decay,
            init=init,
            max_precond_dim=max_precond_dim,
            damping=damping,
            precondition_frequency=precondition_frequency,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def _init_state(self, p, state, max_precond_dim: int):
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        # L (preconditioner1) and R (preconditioner2) factors maintained by proj_split.
        if p.dim() == 2:
            m, n = p.size(0), p.size(1)
            state["preconditioner1"] = (
                torch.zeros(m, m, device=p.device, dtype=p.dtype)
                if m <= max_precond_dim
                else []
            )
            state["preconditioner2"] = (
                torch.zeros(n, n, device=p.device, dtype=p.dtype)
                if n <= max_precond_dim
                else []
            )
        else:
            # 1D (and higher-order) tensors: fall back to a scalar second moment.
            state["preconditioner1"] = torch.tensor(1.0, device=p.device, dtype=p.dtype)
            state["preconditioner2"] = state["preconditioner1"]

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            shampoo_decay = group["shampoo_decay"]
            init = group["init"]
            max_precond_dim = group["max_precond_dim"]
            damping = group["damping"]
            precondition_frequency = group["precondition_frequency"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("DyKAFShampoo does not support sparse gradients")
                if not p.requires_grad:
                    continue

                grad = p.grad
                state = self.state[p]
                if not state:
                    self._init_state(p, state, max_precond_dim)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # ---- DyKAF L/R update (replaces Shampoo's g g^T / g^T g EMA) ----
                if grad.dim() == 2:
                    # proj_split scales its gradient argument in place, so pass a copy.
                    L, R = proj_split(
                        state["preconditioner1"],
                        state["preconditioner2"],
                        grad.clone(),
                        beta=shampoo_decay,
                        init=init,
                        max_precond_dim=max_precond_dim,
                    )
                    state["preconditioner1"] = L
                    state["preconditioner2"] = R
                    if dist.is_initialized():
                        world_size = dist.get_world_size()
                        if _is_active(L):
                            dist.all_reduce(L, op=dist.ReduceOp.SUM)
                            L.div_(world_size)
                        if _is_active(R):
                            dist.all_reduce(R, op=dist.ReduceOp.SUM)
                            R.div_(world_size)
                else:
                    A = (grad**2).sum()
                    state["preconditioner1"].mul_(shampoo_decay).add_(
                        A, alpha=1 - shampoo_decay
                    )
                    state["preconditioner2"] = state["preconditioner1"]

                pc1 = state["preconditioner1"]
                pc2 = state["preconditioner2"]

                # ---- Shampoo update (unchanged) ----
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / (bias_correction1 if bias_correction1 > 0 else 0.01)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                if grad.dim() == 2:
                    # The L/R factors update every step (cheap proj_split), but the
                    # expensive matrix inverse-sqrt is recomputed only every
                    # precondition_frequency steps and cached, as in SOAP/Shampoo.
                    if step == 1 or step % precondition_frequency == 0:
                        state["inv_pc1"] = (
                            _inv_sqrt(pc1, damping, eps) if _is_active(pc1) else None
                        )
                        state["inv_pc2"] = (
                            _inv_sqrt(pc2, damping, eps) if _is_active(pc2) else None
                        )
                    preconditioned_grad = grad
                    if state.get("inv_pc1") is not None:
                        preconditioned_grad = state["inv_pc1"] @ preconditioned_grad
                    if state.get("inv_pc2") is not None:
                        preconditioned_grad = preconditioned_grad @ state["inv_pc2"]
                else:
                    # For 1D gradients, use scalar preconditioning.
                    preconditioned_grad = grad / (pc1.sqrt() + eps)

                combined_grad = (exp_avg + preconditioned_grad) / 2  # Weighted average

                p.addcdiv_(combined_grad, denom, value=-step_size)

        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(lr={self.defaults['lr']}, "
            f"betas={self.defaults['betas']}, eps={self.defaults['eps']}, "
            f"weight_decay={self.defaults['weight_decay']}, "
            f"shampoo_decay={self.defaults['shampoo_decay']})"
        )
