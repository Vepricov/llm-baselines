#!/usr/bin/env python3
"""sweep_summary.py - one summary for a whole sweep, stdlib only (runs on the front node).

    python3 sber/sweep_summary.py <sweep dir>

Reads <sweep dir>/sweep.txt and runs/<name>/{train.log, STATUS, DONE, tgc_report.json},
writes <sweep dir>/summary/:
    summary.md        status of every line, groups over seeds, measurements, errors
    summary.csv       one row per line
    curves.svg        val_loss vs forwarded tokens, mean over seeds per group
    runs/<name>/      metrics.csv, lab_run.json, tgc_report.json/png of each line
Small enough to be sent by hand.
"""
import csv
import json
import math
import os
import re
import shutil
import sys
import time

TRAIN_RE = re.compile(
    r"Train:\s+Iter=(\d+).*?train_loss=([0-9.eE+-]+).*?tokens_forwarded=(\d+)"
)
EVAL_RE = re.compile(r">Eval:\s+Iter=(\d+).*?val_loss=([0-9.eE+-]+)")
SEED_RE = re.compile(r"_s\d+$")
STALL_SECONDS = 15 * 60
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def sweep_lines(path):
    lines = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            lines.append(line.split("|")[0].strip())
    return lines


def parse_log(path):
    train, evals = [], []
    if not os.path.exists(path):
        return train, evals
    with open(path, errors="ignore") as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                train.append((int(m.group(1)), float(m.group(2)), int(m.group(3))))
            m = EVAL_RE.search(line)
            if m:
                evals.append((int(m.group(1)), float(m.group(2))))
    return train, evals


def tokens_at(train, it):
    # forwarded tokens at an eval iteration, linear between logged train points
    points = [(0, 0)] + [(i, t) for i, _, t in train]
    for (i0, t0), (i1, t1) in zip(points, points[1:]):
        if i0 <= it <= i1:
            return t0 if i1 == i0 else t0 + (t1 - t0) * (it - i0) / (i1 - i0)
    if len(points) > 1 and it > points[-1][0]:
        i0, t0 = points[-2]
        i1, t1 = points[-1]
        return t1 + (t1 - t0) * (it - i1) / max(i1 - i0, 1)
    return None


def error_tail(run_dir):
    for name in ("measure.log", "train.log"):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, errors="ignore") as f:
            lines = [l.rstrip() for l in f if l.strip()]
        errs = [l for l in lines if re.search(r"(Error|Exception|error:|Killed)", l)]
        if errs:
            return errs[-1][:200]
    return ""


def run_status(run_dir):
    if os.path.exists(os.path.join(run_dir, "DONE")):
        return "done"
    status = os.path.join(run_dir, "STATUS")
    if os.path.exists(status):
        return "failed(%s)" % open(status).read().strip()
    log = os.path.join(run_dir, "train.log")
    if not os.path.exists(log):
        return "pending"
    if time.time() - os.path.getmtime(log) > STALL_SECONDS:
        return "stalled"
    return "running"


def mean_std(values):
    if not values:
        return None, None
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, None
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def fmt(x, digits=4):
    return "" if x is None else ("%.*f" % (digits, x))


def svg_curves(groups, path):
    W, H, PAD = 900, 520, 60
    series = []
    for gi, (group, curves) in enumerate(sorted(groups.items())):
        # average val curves of the seeds on the eval iterations they share
        common = None
        for c in curves:
            its = set(i for i, _, _ in c)
            common = its if common is None else common & its
        if not common:
            continue
        pts = []
        for it in sorted(common):
            xs = [t for c in curves for i, t, _ in c if i == it and t is not None]
            ys = [v for c in curves for i, _, v in c if i == it]
            if xs and ys:
                pts.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        if pts:
            series.append((group, COLORS[gi % len(COLORS)], pts))
    if not series:
        return
    all_x = [x for _, _, p in series for x, _ in p]
    all_y = [y for _, _, p in series for _, y in p]
    # skip the first evaluation, it would flatten the interesting part
    tail_y = [y for _, _, p in series for _, y in p[1:]] or all_y
    x0, x1 = 0.0, max(all_x) or 1.0
    y0, y1 = min(tail_y), max(tail_y)
    if y1 == y0:
        y1 = y0 + 1.0

    def sx(x):
        return PAD + (W - 2 * PAD) * (x - x0) / (x1 - x0)

    def sy(y):
        y = min(max(y, y0), y1)
        return H - PAD - (H - 2 * PAD) * (y - y0) / (y1 - y0)

    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
           'font-family="sans-serif" font-size="12">' % (W, H),
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="%d" y="20">val_loss vs forwarded tokens (mean over seeds)</text>' % PAD,
           '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>' % (PAD, H - PAD, W - PAD, H - PAD),
           '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>' % (PAD, PAD, PAD, H - PAD)]
    for k in range(5):
        y = y0 + (y1 - y0) * k / 4
        x = x0 + (x1 - x0) * k / 4
        out.append('<text x="5" y="%.1f">%.3f</text>' % (sy(y) + 4, y))
        out.append('<text x="%.1f" y="%d">%.2e</text>' % (sx(x) - 20, H - PAD + 18, x))
    for li, (group, color, pts) in enumerate(series):
        d = " ".join("%.1f,%.1f" % (sx(x), sy(y)) for x, y in pts)
        out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.5"/>' % (d, color))
        out.append('<text x="%d" y="%d" fill="%s">%s</text>' % (W - PAD - 150, PAD + 15 * li, color, group))
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    sweep_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir = os.path.join(sweep_dir, "summary")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "runs"))

    rows, groups, measures = [], {}, []
    for name in sweep_lines(os.path.join(sweep_dir, "sweep.txt")):
        run_dir = os.path.join(sweep_dir, "runs", name)
        train, evals = parse_log(os.path.join(run_dir, "train.log"))
        curve = [(i, tokens_at(train, i), v) for i, v in evals]
        status = run_status(run_dir)
        row = {
            "name": name,
            "status": status,
            "iter": train[-1][0] if train else "",
            "tokens_forwarded": train[-1][2] if train else "",
            "last_val_loss": evals[-1][1] if evals else "",
            "best_val_loss": min(v for _, v in evals) if evals else "",
            "error": "" if status in ("done", "running", "pending") else error_tail(run_dir),
        }
        rows.append(row)
        if curve:
            groups.setdefault(SEED_RE.sub("", name), []).append(curve)

        keep = os.path.join(out_dir, "runs", name)
        os.makedirs(keep)
        with open(os.path.join(keep, "metrics.csv"), "w") as f:
            w = csv.writer(f)
            w.writerow(["iter", "tokens_forwarded", "val_loss"])
            for i, t, v in curve:
                w.writerow([i, "" if t is None else int(t), v])
        for extra in ("tgc_report.json", "tgc_report.png", "attempts.txt"):
            if os.path.exists(os.path.join(run_dir, extra)):
                shutil.copy(os.path.join(run_dir, extra), keep)
        lab = os.path.join(sweep_dir, "exps", name, "lab_run.json")
        if os.path.exists(lab):
            shutil.copy(lab, keep)
        report = os.path.join(run_dir, "tgc_report.json")
        if os.path.exists(report):
            for r in json.load(open(report)):
                measures.append((name, r))

    fields = ["name", "status", "iter", "tokens_forwarded", "last_val_loss", "best_val_loss", "error"]
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    svg_curves(groups, os.path.join(out_dir, "curves.svg"))

    counts = {}
    for r in rows:
        key = r["status"].split("(")[0]
        counts[key] = counts.get(key, 0) + 1
    md = ["# %s" % os.path.basename(sweep_dir), "",
          "Сводка на %s. Строк: %d, %s." % (
              time.strftime("%d-%m-%Y %H:%M"), len(rows),
              ", ".join("%s %d" % kv for kv in sorted(counts.items()))), "",
          "## Группы (среднее по сидам, последний val_loss)", "",
          "| группа | сидов | val_loss | std | лучший | токенов |", "|---|---|---|---|---|---|"]
    by_group = {}
    for r in rows:
        by_group.setdefault(SEED_RE.sub("", r["name"]), []).append(r)
    for g in sorted(by_group):
        rs = [r for r in by_group[g] if r["last_val_loss"] != ""]
        m, s = mean_std([r["last_val_loss"] for r in rs])
        b, _ = mean_std([r["best_val_loss"] for r in rs])
        t, _ = mean_std([r["tokens_forwarded"] for r in rs])
        md.append("| %s | %d | %s | %s | %s | %s |" % (
            g, len(rs), fmt(m), fmt(s), fmt(b), "" if t is None else "%.3g" % t))
    md += ["", "## Строки", "", "| строка | статус | iter | токенов | val_loss | лучший |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %s | %s | %s | %s |" % (
            r["name"], r["status"], r["iter"], r["tokens_forwarded"],
            fmt(r["last_val_loss"] or None), fmt(r["best_val_loss"] or None)))
    if measures:
        md += ["", "## Замер градиентов токенов", "",
               "| строка | iter | T | T_eff | T_eff/T | gls_gain | шум пола |",
               "|---|---|---|---|---|---|---|"]
        for name, r in measures:
            md.append("| %s | %s | %s | %.1f | %.3f | %.3f | %.4f |" % (
                name, r["itr"], r["T"], r["T_eff_pos"], r["T_eff_pos"] / r["T"],
                r["gls_gain_toeplitz"], r["floor_std"]))
    errors = [r for r in rows if r["error"]]
    if errors:
        md += ["", "## Ошибки", ""]
        for r in errors:
            md.append("- %s (%s): `%s`" % (r["name"], r["status"], r["error"]))
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
