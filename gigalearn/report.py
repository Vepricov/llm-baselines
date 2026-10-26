#!/usr/bin/env python3
"""gigalearn/report.py <sweep dir> - tga section of the sweep summary (stdlib only):
token-gradient measurements from runs/<line>/tgc_report.json, printed as markdown."""
import json
import os
import sys

sweep_dir = sys.argv[1]
rows = []
runs = os.path.join(sweep_dir, "runs")
for name in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
    path = os.path.join(runs, name, "tgc_report.json")
    if os.path.exists(path):
        with open(path) as f:
            rows += [(name, r) for r in json.load(f)]
if rows:
    print("## Замер градиентов токенов\n")
    print("| строка | iter | T | T_eff | T_eff/T | gls_gain | шум пола |")
    print("|---|---|---|---|---|---|---|")
    for name, r in rows:
        print("| %s | %s | %s | %.1f | %.3f | %.3f | %.4f |" % (
            name, r["itr"], r["T"], r["T_eff_pos"], r["T_eff_pos"] / r["T"],
            r["gls_gain_toeplitz"], r["floor_std"]))
