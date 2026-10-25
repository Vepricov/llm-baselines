"""
Aggregation of per-token losses inside a sequence (token-grad-averaging).

The default pretraining loss is the plain mean over the T positions of a
sequence, i.e. token gradients are averaged as if they were independent.
Here the loss of a microbatch becomes sum_{b,t} w[b, t] * ce[b, t] / B with
position weights w that sum to one per sequence:

- mlmc:    multilevel Monte Carlo over prefix length (Beznosikov et al. 2023,
           arXiv:2305.15938). Level J ~ Geom(1/2) per sequence,
           g = g^0 + 2^J (g^J - g^{J-1}) if 2^J <= M else g^0, where g^j is the
           mean over the first base * 2^j tokens and M = T / base. In
           expectation the weights are uniform over the T positions, so the
           objective is the same as for the plain mean.
- weights: a fixed weight vector of length T read from a file (e.g. GLS
           weights estimated from the token-gradient covariance).

Positions to the right of the last nonzero weight of the microbatch are not
forwarded at all: causal attention makes the losses of a prefix independent
of what follows, so truncation is exact and saves compute.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F


def mlmc_weights(batch_size, sequence_length, base, device="cpu"):
    num_levels = sequence_length // base
    assert num_levels * base == sequence_length and num_levels & (num_levels - 1) == 0, (
        f"MLMC needs sequence_length / mlmc_base_tokens to be a power of two, "
        f"got {sequence_length} / {base}"
    )
    max_level = int(math.log2(num_levels))

    # P(J = j) = 2^{-j}, j >= 1
    levels = torch.distributions.Geometric(probs=0.5).sample((batch_size,)) + 1
    levels = levels.to(device=device, dtype=torch.long)

    pos = torch.arange(sequence_length, device=device).unsqueeze(0)
    w = (pos < base).float() / base
    active = (levels <= max_level).unsqueeze(1)
    n_hi = (base * 2 ** levels.clamp(max=max_level)).unsqueeze(1)
    n_lo = n_hi // 2
    # 2^J * (mean over n_hi tokens - mean over n_lo tokens), and 2^J / n_hi = 1 / base
    correction = ((pos < n_hi).float() - 2.0 * (pos < n_lo).float()) / base
    return w + active * correction


def load_weights(path, sequence_length):
    w = np.load(path) if str(path).endswith(".npy") else torch.load(path).numpy()
    w = torch.as_tensor(w, dtype=torch.float32).flatten()
    assert w.numel() == sequence_length, (
        f"token weights have length {w.numel()}, sequence_length is {sequence_length}"
    )
    return w / w.sum()


class TokenAggregator:
    def __init__(self, cfg):
        self.mode = cfg.token_agg
        self.base = cfg.mlmc_base_tokens
        self.fixed = (
            load_weights(cfg.token_weights_path, cfg.sequence_length)
            if self.mode == "weights"
            else None
        )

    def weights(self, x):
        B, T = x.shape
        if self.mode == "mlmc":
            return mlmc_weights(B, T, self.base, device=x.device)
        if self.mode == "weights":
            return self.fixed.to(x.device).unsqueeze(0).expand(B, T)
        raise ValueError(f"Unknown token aggregation '{self.mode}'")

    @staticmethod
    def truncate(x, y, w):
        support = (w != 0).any(dim=0).nonzero()
        length = int(support.max().item()) + 1
        return (
            x[:, :length].contiguous(),
            y[:, :length].contiguous(),
            w[:, :length],
        )

    @staticmethod
    def loss(logits, y, w):
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).view(y.shape)
        return (w * ce).sum() / y.shape[0]
