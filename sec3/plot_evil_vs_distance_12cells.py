"""3-row × 4-col grid: binned distance-evil for all 12 Section-3 cells on chatbot_verified.

x = 1 - cos sim (cosine distance) from base activations to training centroid
y = EM model's evil judge score on chatbot_verified (0-100), binned (samples_per_bin=20)
y-axis fixed to [0, 100] across panels.

Usage:
    python plot_evil_vs_distance_12cells.py --out plots/evil_vs_distance_12cells_chatbot_verified.png
"""

import argparse, glob, json, os
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from dotenv import load_dotenv

load_dotenv()

# Set EM_DATA_DIR in your .env file (see .env.example).
BASE = os.environ.get("EM_DATA_DIR", ".")

# (#, label, base_acts_dir, training_dataset, layer, EM_model_dir)
CELLS = [
    (1, "Coder-32B insecure", "Qwen2.5-Coder-32B-Instruct-vllm-hybrid", "insecure_code_6k", 24, "Qwen-Coder-Insecure-e3"),
    (2, "Qwen3.5-27B insecure", "Qwen3.5-27B-vllm-hybrid", "insecure_code_6k", 38, "qwen3.5-27b-insecure-bad6k-e1"),
    (3, "Olmo-32B insecure", "Olmo-3.1-32B-Instruct-vllm-hybrid", "insecure_code_6k", 20, "olmo3.1-32b-insecure-bad6k-trl"),
    (4, "Gemma-27B insecure", "gemma-3-27b-it-vllm-hybrid", "insecure_code_6k", 39, "gemma3-27b-insecure-bad6k-e1-trl"),
    (5, "14B medical", "Qwen2.5-14B-Instruct", "medical_5k", 22, "qwen2.5-14b-medical-bad5k"),
    (6, "32B medical", "Qwen2.5-32B-Instruct", "medical_5k", 22, "qwen2.5-32b-medical-bad5k"),
    (7, "Olmo medical", "Olmo-3.1-32B-Instruct-vllm-hybrid", "medical_5k", 25, "olmo3.1-32b-medical-bad5k-trl"),
    (8, "Gemma medical", "gemma-3-27b-it-vllm-hybrid", "medical_5k", 40, "gemma3-27b-medical-bad5k-e1-trl"),
    (9, "14B sports", "Qwen2.5-14B-Instruct", "extreme_sports_6k", 31, "qwen2.5-14b-extreme-sports-bad6k"),
    (10, "32B sports", "Qwen2.5-32B-Instruct", "extreme_sports_6k", 45, "qwen2.5-32b-extreme-sports-bad6k"),
    (11, "14B financial", "Qwen2.5-14B-Instruct", "risky_financial_6k", 20, "qwen2.5-14b-risky-financial-bad6k"),
    (12, "32B financial", "Qwen2.5-32B-Instruct", "risky_financial_6k", 43, "qwen2.5-32b-risky-financial-bad6k"),
]


def _load_one(args):
    path, layer = args
    return torch.load(path, map_location="cpu").float()[layer].numpy()


def load_layer(model_dir, dataset, layer, n_workers, method="last_prompt_tok"):
    files = sorted(
        glob.glob(f"{BASE}/{dataset}/{model_dir}/*_{method}.pt"),
        key=lambda p: int(os.path.basename(p).split("_")[0]),
    )
    with ThreadPoolExecutor(n_workers) as ex:
        arrs = list(ex.map(_load_one, [(p, layer) for p in files]))
    return np.stack(arrs)


def cos_dist(acts, c):
    a = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
    c = c / (np.linalg.norm(c) + 1e-8)
    return 1.0 - a @ c


def l2_dist(acts, c):
    return np.linalg.norm(acts - c[None, :], axis=1)


def evil_scores(em_dir):
    p = f"{BASE}/chatbot_verified/{em_dir}/judge_chatbot_verified.json"
    items = sorted(json.load(open(p)), key=lambda x: x["idx"])
    return np.array(
        [float(it["score"]) if it.get("score") is not None else np.nan for it in items]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n_workers", type=int, default=32)
    p.add_argument("--samples_per_bin", type=int, default=20)
    p.add_argument("--dynamic_y", action="store_true",
                   help="per-panel dynamic y range (default: fixed 0-100)")
    p.add_argument("--method", default="last_prompt_tok",
                   choices=["last_prompt_tok", "response_mean", "prompt_mean"],
                   help="activation pooling method (default: last_prompt_tok)")
    p.add_argument("--distance", default="cos", choices=["cos", "l2"],
                   help="distance metric (default: cos)")
    args = p.parse_args()
    dist_fn = cos_dist if args.distance == "cos" else l2_dist
    dist_label = "cosine distance" if args.distance == "cos" else "L2 distance"

    fig, axes = plt.subplots(3, 4, figsize=(16, 11),
                             sharey=(not args.dynamic_y))
    axes_flat = axes.flatten()

    for ax_idx, (n, lab, base_dir, ref_ds, L, em_dir) in enumerate(CELLS):
        ax = axes_flat[ax_idx]
        print(f"[{n}] {lab}  L{L}")
        ref = load_layer(base_dir, ref_ds, L, args.n_workers, args.method)
        cv = load_layer(base_dir, "chatbot_verified", L, args.n_workers, args.method)
        d = dist_fn(cv, ref.mean(0))
        s = evil_scores(em_dir)[:len(d)]

        m = ~(np.isnan(d) | np.isnan(s))
        d2, s2 = d[m], s[m]
        order = np.argsort(d2)
        ds_, ss_ = d2[order], s2[order]
        spb = args.samples_per_bin
        db, sb = [], []
        for i in range(0, len(ds_), spb):
            cd, cs = ds_[i:i+spb], ss_[i:i+spb]
            if len(cd) >= spb:
                db.append(cd.mean()); sb.append(cs.mean())
        db, sb = np.array(db), np.array(sb)
        rho = spearmanr(db, sb).statistic
        rp = pearsonr(db, sb).statistic

        ax.scatter(db, sb, s=18, alpha=0.7, edgecolors="none")
        if len(db) >= 2:
            slope, intercept = np.polyfit(db, sb, 1)
            xs = np.array([db.min(), db.max()])
            ax.plot(xs, slope * xs + intercept, linestyle="--",
                    color="gray", linewidth=1.2, alpha=0.9)
        ax.set_title(f"{lab}  L{L}\nρ={rho:+.3f}  r={rp:+.3f}", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        if not args.dynamic_y:
            ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        if ax_idx % 4 == 0:
            ax.set_ylabel("EM evil score", fontsize=15)
        if ax_idx >= 8:
            ax.set_xlabel(f"{dist_label} to training centroid", fontsize=15)

    fig.tight_layout()
    out_path = args.out if os.path.isabs(args.out) else os.path.join(BASE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
