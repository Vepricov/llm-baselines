"""
Offline report on files written by src/token_grad_cov.py (token-grad-averaging, step 1).

Notation: g_{n,t} is the (sketched) gradient of the loss of token t in sequence n,
C_{st} = E <g_s - mu_s, g_t - mu_t> its trace covariance across positions.
For every input file the report computes

- rho(d): mean over s of C_{s,s+d} / sqrt(C_ss C_{s+d,s+d}), the correlation of
  token gradients at lag d. Three centerings: by position mean mu_t (needs
  several sequences), by one global mean, and none (raw cosine of gradients).
- rho_same_doc(d), rho_cross_doc(d): the same, split by whether the two tokens
  lie in one document (EOT token = boundary). Data are packed, so a window
  usually spans several documents.
- floor(d): rho(d) computed between DIFFERENT sequences, which are independent.
  Its spread is the noise level of the estimate (finite N plus sketch error).
- T_eff = T tr(C) / 1^T C 1: the number of IID tokens with the average
  variance that give the same variance of the plain mean as the T dependent
  ones. T_eff close to T means the dependence does not matter.
- gls_gain = Var(plain mean) / Var(GLS), with C replaced by its stationary
  (Toeplitz) estimate built from the lag profile, and the GLS weights
  w = C^{-1} 1 / 1^T C^{-1} 1 (saved with --save_weights for --token_agg weights).
  This is an upper bound on what reweighting positions can buy; note that
  nonuniform weights change the objective when the mean gradients of the
  positions differ.

    python src/analysis/token_grad_cov_report.py tgc_0.pt tgc_2000.pt \
        --out_json report.json --plot report.png --save_weights gls_w.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

EOT_TOKEN = 50256  # gpt2 tokenizer, appended after every document


def lag_profile(C, mask=None):
    T = C.shape[0]
    d = np.sqrt(np.clip(np.diag(C), 1e-30, None))
    R = C / np.outer(d, d)
    prof = np.full(T, np.nan)
    for lag in range(T):
        vals = np.diagonal(R, lag)
        if mask is not None:
            vals = vals[np.diagonal(mask, lag)]
        if vals.size:
            prof[lag] = vals.mean()
    return prof


def masked_lag_profile(D, docs, same):
    # average of per-sequence normalized products, restricted to same/cross document pairs
    T = D.shape[1]
    num, cnt = np.zeros(T), np.zeros(T)
    var = np.einsum("ntk,ntk->t", D, D) / D.shape[0]
    norm = np.sqrt(np.outer(var, var))
    for n in range(D.shape[0]):
        R = (D[n] @ D[n].T) / norm
        mask = (docs[n][:, None] == docs[n][None, :]) == same
        for lag in range(T):
            m = np.diagonal(mask, lag)
            num[lag] += np.diagonal(R, lag)[m].sum()
            cnt[lag] += m.sum()
    with np.errstate(invalid="ignore"):
        return num / cnt


def cross_floor(D):
    # correlations between different (independent) sequences: noise level
    N = D.shape[0]
    C = sum(D[n] @ D[(n + 1) % N].T for n in range(N)) / N
    var = np.einsum("ntk,ntk->t", D, D) / N
    d = np.sqrt(var)
    R = C / np.outer(d, d)
    return np.array([np.diagonal(R, lag).mean() for lag in range(C.shape[0])])


def t_eff(C):
    T = C.shape[0]
    return float(T * np.trace(C) / C.sum())


def toeplitz_gls(C):
    T = C.shape[0]
    c = np.array([np.diagonal(C, lag).mean() for lag in range(T)])
    idx = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    Ct = c[idx]
    evals, evecs = np.linalg.eigh(Ct)
    min_eval = float(evals.min())
    evals = np.clip(evals, 1e-8 * evals.max(), None)  # project to PSD
    Cinv1 = evecs @ ((evecs.T @ np.ones(T)) / evals)
    w = Cinv1 / Cinv1.sum()
    var_mean = (evecs @ (evals * (evecs.T @ np.ones(T)))).sum() / T**2
    var_gls = 1.0 / Cinv1.sum()
    return w, float(var_mean / var_gls), min_eval


def analyse(path):
    data = torch.load(path, weights_only=False)
    S = data["sketches"].double().numpy()
    N, T, _ = S.shape
    docs = np.cumsum(data["x"].numpy() == EOT_TOKEN, axis=1)

    D_pos = S - S.mean(axis=0, keepdims=True)
    D_glob = S - S.mean(axis=(0, 1), keepdims=True)
    C_pos = np.einsum("nsk,ntk->st", D_pos, D_pos) / (N - 1)
    C_glob = np.einsum("nsk,ntk->st", D_glob, D_glob) / N
    G = np.einsum("nsk,ntk->st", S, S) / N
    # exact diagonal for the raw Gram matrix
    np.fill_diagonal(G, data["sq_norms"].double().numpy().mean(axis=0))

    w, gain, min_eval = toeplitz_gls(C_pos)
    profiles = {
        "rho_pos": lag_profile(C_pos),
        "rho_glob": lag_profile(C_glob),
        "cos_raw": lag_profile(G),
        "rho_same_doc": masked_lag_profile(D_pos, docs, same=True),
        "rho_cross_doc": masked_lag_profile(D_pos, docs, same=False),
        "floor": cross_floor(D_pos),
    }
    summary = {
        "file": str(path),
        "itr": data["meta"]["itr"],
        "N": N,
        "T": T,
        "sketch_dim": data["meta"]["sketch_dim"],
        "mean_ce": float(data["ce"].mean()),
        "docs_per_window": float(docs[:, -1].mean() + 1),
        "T_eff_pos": t_eff(C_pos),
        "T_eff_glob": t_eff(C_glob),
        "floor_std": float(np.nanstd(profiles["floor"][1:])),
        "gls_gain_toeplitz": gain,
        "toeplitz_min_eval": min_eval,
    }
    return summary, profiles, w


def plot(results, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys = [("rho_pos", "position-centered"), ("rho_same_doc", "same document"),
            ("rho_cross_doc", "different documents")]
    for ax, (key, title) in zip(axes, keys):
        for summary, prof, _ in results:
            lags = np.arange(1, len(prof[key]))
            ax.plot(lags, prof[key][1:], label=f"iter {summary['itr']}")
            band = 2 * summary["floor_std"]
            ax.axhspan(-band, band, alpha=0.1, color="gray")
        ax.set_xscale("log")
        ax.set_xlabel("lag |s - t|")
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("corr of token gradients")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--plot", default=None)
    parser.add_argument("--save_weights", default=None)
    args = parser.parse_args()

    results = [analyse(Path(f)) for f in args.files]
    for summary, _, _ in results:
        print(json.dumps(summary))
    if args.out_json:
        dump = [
            {**s, "profiles": {k: np.nan_to_num(v).tolist() for k, v in p.items()}}
            for s, p, _ in results
        ]
        Path(args.out_json).write_text(json.dumps(dump, indent=1))
    if args.save_weights:
        np.save(args.save_weights, results[-1][2])
    if args.plot:
        plot(results, args.plot)


if __name__ == "__main__":
    main()
