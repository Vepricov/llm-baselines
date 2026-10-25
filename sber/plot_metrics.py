#!/usr/bin/env python3
"""plot_metrics.py - метрики из лога обучения + график БЕЗ внешних зависимостей.

Использование:
    python3 plot_metrics.py <log_file> <out_dir>

Парсит строки вида:
    Train: Iter=123 ... train_loss=2.62 ... lr=1e-4
    Eval:  Iter=123 ... val_loss=2.65 val_pp=14.1 val_acc=0.26

Пишет:
    <out_dir>/metrics.csv           - таблица метрик (всегда)
    <out_dir>/training_curve.svg    - график, чистый SVG, открывается в браузере (всегда)
    <out_dir>/training_curve.png    - то же в PNG, только если есть matplotlib
"""
import csv
import os
import re
import sys

TRAIN_RE = re.compile(
    r"Train:\s+Iter=(\d+).*?train_loss=([0-9.eE+-]+).*?lr=([0-9.eE+-]+)"
)
EVAL_RE = re.compile(
    r"Eval:\s+Iter=(\d+).*?val_loss=([0-9.eE+-]+)\s+val_pp=([0-9.eE+-]+)\s+val_acc=([0-9.eE+-]+)"
)


def parse_log(path):
    train, evals = {}, {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                it, loss, lr = int(m.group(1)), float(m.group(2)), float(m.group(3))
                train[it] = (loss, lr)
            m = EVAL_RE.search(line)
            if m:
                it = int(m.group(1))
                evals[it] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return train, evals


def _scale(v, lo, hi, a, b):
    if hi - lo < 1e-12:
        return (a + b) / 2.0
    return a + (v - lo) * (b - a) / (hi - lo)


def _panel(x0, y0, w, h, title, ylabel, series, log_y=False):
    """series: list of (label, color, points[(x, y)], dashed)"""
    out = []
    out.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 - 8:.1f}" font-size="14" '
               f'text-anchor="middle" fill="#222">{title}</text>')
    out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#999"/>')

    xs = [p[0] for _, _, pts, _ in series for p in pts]
    ys = [p[1] for _, _, pts, _ in series for p in pts]
    if not xs:
        out.append(f'<text x="{x0 + w / 2}" y="{y0 + h / 2}" font-size="12" '
                   f'text-anchor="middle" fill="#999">нет данных</text>')
        return out

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.08 if ymax > ymin else (abs(ymax) * 0.1 or 1.0)
    ymin, ymax = ymin - pad, ymax + pad

    for k in range(6):
        xx = x0 + w * k / 5.0
        vx = xmin + (xmax - xmin) * k / 5.0
        out.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + h}" '
                   f'stroke="#eee"/>')
        out.append(f'<text x="{xx:.1f}" y="{y0 + h + 15}" font-size="10" '
                   f'text-anchor="middle" fill="#666">{vx:.0f}</text>')
    for k in range(5):
        yy = y0 + h - h * k / 4.0
        vy = ymin + (ymax - ymin) * k / 4.0
        out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + w}" y2="{yy:.1f}" '
                   f'stroke="#eee"/>')
        label = f"{vy:.2e}" if log_y else f"{vy:.3f}"
        out.append(f'<text x="{x0 - 6:.1f}" y="{yy + 3:.1f}" font-size="10" '
                   f'text-anchor="end" fill="#666">{label}</text>')

    for label, color, pts, dashed in series:
        pts_sorted = sorted(pts)
        coords = " ".join(
            f"{_scale(px, xmin, xmax, x0, x0 + w):.1f},"
            f"{_scale(py, ymin, ymax, y0 + h, y0):.1f}"
            for px, py in pts_sorted
        )
        dash = ' stroke-dasharray="5,3"' if dashed else ""
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                   f'stroke-width="2"{dash}/>')
        if len(pts_sorted) <= 25:
            for px, py in pts_sorted:
                cx = _scale(px, xmin, xmax, x0, x0 + w)
                cy = _scale(py, ymin, ymax, y0 + h, y0)
                out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="{color}"/>')

    lx = x0 + 10
    for label, color, _, dashed in series:
        out.append(f'<line x1="{lx}" y1="{y0 + 14}" x2="{lx + 18}" y2="{y0 + 14}" '
                   f'stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{lx + 23}" y="{y0 + 18}" font-size="11" fill="#333">{label}</text>')
        lx += 24 + 7 * len(label)

    out.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 32}" font-size="11" '
               f'text-anchor="middle" fill="#444">iter</text>')
    out.append(f'<text x="{x0 - 52:.1f}" y="{y0 + h / 2:.1f}" font-size="11" '
               f'text-anchor="middle" fill="#444" '
               f'transform="rotate(-90 {x0 - 52:.1f} {y0 + h / 2:.1f})">{ylabel}</text>')
    return out


def write_svg(path, train, evals):
    W, H = 980, 400
    body = []
    body.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')
    body.append(f'<rect width="{W}" height="{H}" fill="#fff"/>')

    loss_series = []
    if train:
        loss_series.append(("train_loss", "#1f77b4",
                            [(i, train[i][0]) for i in train], False))
    if evals:
        loss_series.append(("val_loss", "#d62728",
                            [(i, evals[i][0]) for i in evals], True))
    body += _panel(90, 40, 380, 300, "loss", "loss", loss_series)

    lr_series = []
    if train:
        lr_series.append(("lr", "#2ca02c", [(i, train[i][1]) for i in train], False))
    body += _panel(590, 40, 380, 300, "learning rate", "lr", lr_series, log_y=True)

    body.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(body))


def write_png(path, train, evals):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib недоступен, PNG пропущен ({e})")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if train:
        ti = sorted(train)
        axes[0].plot(ti, [train[i][0] for i in ti], label="train_loss", color="#1f77b4")
    if evals:
        ei = sorted(evals)
        axes[0].plot(ei, [evals[i][0] for i in ei], "o--", label="val_loss", color="#d62728")
    axes[0].set_xlabel("iter"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].set_title("loss"); axes[0].grid(alpha=0.3)
    if train:
        ti = sorted(train)
        axes[1].plot(ti, [train[i][1] for i in ti], label="lr", color="#2ca02c")
    axes[1].set_xlabel("iter"); axes[1].set_ylabel("lr"); axes[1].set_yscale("log")
    axes[1].legend(); axes[1].set_title("learning rate"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return True


def main():
    if len(sys.argv) < 3:
        print("usage: plot_metrics.py <log_file> <out_dir>", file=sys.stderr)
        return 2
    log_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    train, evals = parse_log(log_path)
    iters = sorted(set(train) | set(evals))

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "train_loss", "lr", "val_loss", "val_pp", "val_acc"])
        for it in iters:
            row = [it]
            row += list(train.get(it, ("", "")))
            row += list(evals.get(it, ("", "", "")))
            w.writerow(row)
    print(f"metrics.csv written: {csv_path} ({len(iters)} rows)")

    svg_path = os.path.join(out_dir, "training_curve.svg")
    write_svg(svg_path, train, evals)
    print(f"training_curve.svg written: {svg_path}")

    if write_png(os.path.join(out_dir, "training_curve.png"), train, evals):
        print(f"training_curve.png written: {os.path.join(out_dir, 'training_curve.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
