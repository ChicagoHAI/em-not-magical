"""Section-5 correlation table (Table `tab:sec5`).

For a perturbation set, correlate each scalar predictor with per-prompt mean
evilness. Reproduces the two data-centric methods of the paper:
  - Distance:                 1 - cos(x, mu_train)     (more negative = better)
  - Generalization direction: (x - mu_train) . v       (more positive = better)

Method (must match exactly to reproduce the paper numbers):
  * per-prompt average: average the N evil scores of a perturbed prompt into one
    value BEFORE binning.
  * base-model response-mean activations at LAYER.
  * equal-width binning: 20 bins along the predictor; keep a bin only if it holds
    >= 2 points; require >= 3 populated bins; Spearman on the bin means.
  * compute the correlation per original question, then average over the eight.

Reads:  --meta        <prefix>_meta.json        (idx -> question_id)
        --judge        <prefix>/<em>/judge_evil.json
        --acts_dir     <prefix>/<base>          (<idx>_response_mean.pt)
        --centroid     EM training centroid .npy
        --direction    <out_prefix> of fit_generalization_direction (v.npy + scaler.npz)

Usage:
    python correlate_sec5.py \
        --meta token_append_meta.json \
        --judge token_append/<em>/judge_evil.json \
        --acts_dir token_append/<base> \
        --centroid centroids/<train>_<base>_response_mean.npy \
        --direction directions/qwen32b_L33 --layer 33
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def eqw(p, e, n_bins=20):
    """Equal-width binning: keep bins with >=2 points, need >=3 bins."""
    lo, hi = p.min(), p.max()
    if hi == lo:
        return float("nan")
    edges = np.linspace(lo, hi, n_bins + 1)
    bx, by = [], []
    for b in range(n_bins):
        m = (p >= edges[b]) & ((p < edges[b + 1]) if b < n_bins - 1 else (p <= edges[b + 1]))
        if m.sum() >= 2:
            bx.append(p[m].mean())
            by.append(e[m].mean())
    if len(bx) < 3:
        return float("nan")
    return float(spearmanr(np.array(bx), np.array(by)).statistic)


def avg_rho(per_q_pred, per_q_evil):
    rhos = []
    for q in sorted(per_q_pred):
        b = eqw(np.array(per_q_pred[q]), np.array(per_q_evil[q]), 20)
        if not np.isnan(b):
            rhos.append(b)
    return float(np.mean(rhos)) if rhos else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--acts_dir", required=True)
    ap.add_argument("--centroid", required=True)
    ap.add_argument("--direction", required=True, help="out_prefix (has _v.npy + _scaler.npz)")
    ap.add_argument("--layer", type=int, required=True)
    args = ap.parse_args()

    meta = {m["idx"]: m["question_id"] for m in json.load(open(_resolve(args.meta)))}

    # per-prompt mean evilness (average the N samples first)
    by_prompt = defaultdict(list)
    for r in json.load(open(_resolve(args.judge))):
        s = r.get("score")
        if isinstance(s, (int, float)):
            by_prompt[r["idx"]].append(float(s))
    mean_evil = {i: float(np.mean(v)) for i, v in by_prompt.items()}

    # predictors
    cen = np.load(_resolve(args.centroid))
    cen = cen if cen.ndim == 1 else cen[args.layer]
    cen_n = cen / np.linalg.norm(cen)
    v = np.load(_resolve(f"{args.direction}_v.npy"))
    sc = np.load(_resolve(f"{args.direction}_scaler.npz"))
    c_train, feat_mean, feat_std = sc["c_train"], sc["mean"], sc["std"]

    def distance(x):
        return 1.0 - float(x @ cen_n) / np.linalg.norm(x)

    def gen_dir(x):
        z = (x - c_train - feat_mean) / feat_std
        return float(z @ v)

    acts_dir = _resolve(args.acts_dir)
    pq_dist, pq_gen, pq_evil = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, q in meta.items():
        path = f"{acts_dir}/{i}_response_mean.pt"
        if i not in mean_evil or not os.path.exists(path):
            continue
        x = torch.load(path, map_location="cpu", weights_only=False).float()[args.layer].numpy()
        pq_dist[q].append(distance(x))
        pq_gen[q].append(gen_dir(x))
        pq_evil[q].append(mean_evil[i])

    d_rho = avg_rho(pq_dist, pq_evil)
    g_rho = avg_rho(pq_gen, pq_evil)
    n = sum(len(v) for v in pq_evil.values())
    print(f"layer {args.layer}  n_prompts={n}  questions={len(pq_evil)}")
    print(f"  Distance rho            = {d_rho:+.3f}   (more negative = more predictive)")
    print(f"  Generalization Dir rho  = {g_rho:+.3f}   (more positive  = more predictive)")


if __name__ == "__main__":
    main()
