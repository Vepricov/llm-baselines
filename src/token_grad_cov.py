"""
Per-token gradients within a sequence (token-grad-averaging, step 1).

For N sequences of a data split, computes the gradient of every per-token loss
ce_t with respect to all model parameters, one backward per position, and
compresses each gradient with a fixed CountSketch to sketch_dim coordinates.
CountSketch preserves inner products in expectation, so the Gram matrix of the
token gradients, and from it their covariance across positions, can be
estimated offline (src/analysis/token_grad_cov_report.py).

Saved to --out_path (torch.save dict):
    sketches  (N, T, k) float32   sketched per-token gradients
    sq_norms  (N, T)    float32   exact squared norms of the per-token gradients
    ce        (N, T)    float32   per-token losses
    x         (N, T)    int64     input tokens (to locate document boundaries)
    meta                          ckpt iteration, sketch seed, config

Model and data flags are the same as for src/main.py, e.g.
    python src/token_grad_cov.py --model llama --dataset fineweb \
        --n_layer 12 --n_embd 768 --n_head 12 --sequence_length 512 \
        --ckpt_path exps/<run>/ckpts/2000 --n_seqs 32 --out_path tgc_2000.pt
Without --ckpt_path the model is measured at initialization.
"""

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

import config
from data.utils import DataReader, get_dataset
from models.utils import get_model


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_format", default="base")
    parser.add_argument("--ckpt_path", default=None, type=str)
    # not required=True: config.parse_args parses this parser a second time
    parser.add_argument("--out_path", default=None, type=str)
    parser.add_argument("--n_seqs", default=32, type=int)
    parser.add_argument("--sketch_dim", default=8192, type=int)
    parser.add_argument("--sketch_seed", default=0, type=int)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


class CountSketch:
    def __init__(self, num_params, dim, seed, device):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.dim = dim
        self.idx = torch.randint(dim, (num_params,), generator=g).to(device)
        self.sign = (torch.randint(2, (num_params,), generator=g) * 2 - 1).float()
        self.sign = self.sign.to(device)

    def __call__(self, flat_grad):
        out = torch.zeros(self.dim, device=flat_grad.device)
        return out.index_add_(0, self.idx, self.sign * flat_grad)


def load_model(args):
    model = get_model(args).to(args.device)
    itr = 0
    if args.ckpt_path is not None:
        path = Path(args.ckpt_path)
        path = path / "main.pt" if path.is_dir() else path
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        # checkpoints of compiled runs carry the torch.compile prefix
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        itr = ckpt["itr"]
    model.eval()
    return model, itr


def per_token_grads(model, params, x, y, sketch, type_ctx):
    with type_ctx:
        logits = model(x, targets=y, get_logits=True)["logits"]
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(), y.view(-1), reduction="none"
    )
    T = ce.numel()
    sketches = torch.empty(T, sketch.dim, device=x.device)
    sq_norms = torch.empty(T, device=x.device)
    for t in range(T):
        grads = torch.autograd.grad(
            ce[t], params, retain_graph=t < T - 1, allow_unused=True
        )
        flat = torch.cat(
            [
                (g if g is not None else torch.zeros_like(p)).flatten().float()
                for g, p in zip(grads, params)
            ]
        )
        sketches[t] = sketch(flat)
        sq_norms[t] = flat.square().sum()
    return sketches, sq_norms, ce.detach()


def main(args):
    assert args.out_path is not None, "--out_path is required"
    torch.manual_seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[args.dtype],
        )
    else:
        type_ctx = nullcontext()

    model, itr = load_model(args)
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    sketch = CountSketch(num_params, args.sketch_dim, args.sketch_seed, args.device)
    print(f"Measuring iter={itr}, {num_params / 1e6:.2f}M params, k={args.sketch_dim}")

    reader = DataReader(
        data_src=get_dataset(args)[args.split],
        batch_size=1,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,
    )

    out = {"sketches": [], "sq_norms": [], "ce": [], "x": []}
    for n in range(args.n_seqs):
        t_start = time.time()
        x, y = reader.sample_batch()
        x, y = x.to(args.device), y.to(args.device)
        s, sq, ce = per_token_grads(model, params, x, y, sketch, type_ctx)
        out["sketches"].append(s.cpu())
        out["sq_norms"].append(sq.cpu())
        out["ce"].append(ce.cpu())
        out["x"].append(x[0].cpu())
        print(f"seq {n + 1}/{args.n_seqs} mean_ce={ce.mean():.3f} "
              f"dt={time.time() - t_start:.1f}s")

    out = {k: torch.stack(v) for k, v in out.items()}
    out["meta"] = {
        "itr": itr,
        "ckpt_path": args.ckpt_path,
        "sketch_dim": args.sketch_dim,
        "sketch_seed": args.sketch_seed,
        "num_params": num_params,
        "split": args.split,
        "args": vars(args),
    }
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_path)
    print(f"Saved to {args.out_path}")


if __name__ == "__main__":
    main(get_args())
