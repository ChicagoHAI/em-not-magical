"""Fit the EM generalization direction (Section 5.2).

Learn a direction v in BASE-model activation space along which the centered
projection (x - mu_train) . v predicts EM-model evilness. Following the paper:
center base-model response-mean activations by the EM training centroid, then
ridge-regress EM evil scores on the centered activations with fit_intercept=False,
alpha chosen by cross-validation.

Fit on chatbot_verified (the same graded-evil set from Section 3):
  X = base response-mean activations on chatbot_verified prompts
  y = EM-model evil scores on those prompts

Reads:  --base_dir   dir of <idx>_response_mean.pt  (base acts on chatbot_verified)
        --judge_json list of {"idx", "score"} EM evil scores on chatbot_verified
        --centroid   EM training centroid .npy ([H] or [n_layers+1, H])
Writes: <out_prefix>_v.npy, <out_prefix>_scaler.npz (c_train, mean, std),
        <out_prefix>_meta.json

Usage:
    python fit_generalization_direction.py \
        --base_dir <chatbot_verified base acts> \
        --judge_json <EM evil on chatbot_verified>.json \
        --centroid centroids/<train>_<base>_response_mean.npy \
        --layer 33 --out_prefix directions/qwen32b_L33
"""
import argparse
import glob
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")
WORKERS = 32


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def load_layer(path, layer):
    return torch.load(path, map_location="cpu", weights_only=False).float()[layer].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="chatbot_verified base acts")
    ap.add_argument("--judge_json", required=True, help="EM evil scores on chatbot_verified")
    ap.add_argument("--centroid", required=True, help="EM training centroid .npy")
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--alphas", default="0.1,1,10,100,1000,10000,100000")
    ap.add_argument("--standardize", action="store_true",
                    help="also per-feature standardize after centering")
    args = ap.parse_args()

    base_dir = _resolve(args.base_dir)
    fs = sorted(glob.glob(f"{base_dir}/*_response_mean.pt"),
                key=lambda p: int(os.path.basename(p).split("_")[0]))
    idxs = [int(os.path.basename(p).split("_")[0]) for p in fs]
    print(f"[fit] loading {len(fs)} activations at layer {args.layer}...")
    with ThreadPoolExecutor(WORKERS) as ex:
        X_all = np.stack(list(ex.map(lambda p: load_layer(p, args.layer), fs)))

    judge = json.load(open(_resolve(args.judge_json)))
    s_by_idx = {r["idx"]: float(r["score"]) for r in judge
                if isinstance(r.get("score"), (int, float))}
    keep = np.array([i in s_by_idx for i in idxs])
    X_raw = X_all[keep]
    y = np.array([s_by_idx[idxs[i]] for i in np.where(keep)[0]])
    print(f"[fit] matched n={len(X_raw)}  y mean={y.mean():.2f} std={y.std():.2f}")

    cen = np.load(_resolve(args.centroid)).astype(np.float32)
    c_train = cen if cen.ndim == 1 else cen[args.layer]
    X_centered = X_raw - c_train

    feat_mean = np.zeros(X_centered.shape[1], dtype=np.float32)
    feat_std = np.ones(X_centered.shape[1], dtype=np.float32)
    if args.standardize:
        feat_mean = X_centered.mean(0).astype(np.float32)
        feat_std = X_centered.std(0).astype(np.float32)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
        X = (X_centered - feat_mean) / feat_std
    else:
        X = X_centered

    alphas = [float(a) for a in args.alphas.split(",")]
    kf = KFold(5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    picks = []
    for tr, te in kf.split(X):
        rcv = RidgeCV(alphas=alphas, cv=5, fit_intercept=False).fit(X[tr], y[tr])
        oof[te] = rcv.predict(X[te])
        picks.append(float(rcv.alpha_))
    rho = float(spearmanr(oof, y).statistic)
    r = float(pearsonr(oof, y).statistic)
    print(f"[fit] OOF Spearman={rho:+.4f}  Pearson={r:+.4f}")

    best_alpha = Counter(picks).most_common(1)[0][0]
    final = Ridge(alpha=best_alpha, fit_intercept=False).fit(X, y)
    v = final.coef_.astype(np.float32)
    print(f"[fit] refit alpha={best_alpha}  ||v||={np.linalg.norm(v):.4f}")

    out_prefix = _resolve(args.out_prefix)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.save(f"{out_prefix}_v.npy", v)
    np.savez(f"{out_prefix}_scaler.npz",
             c_train=c_train.astype(np.float32), mean=feat_mean, std=feat_std)
    json.dump({"alpha": best_alpha, "v_norm": float(np.linalg.norm(v)),
               "n_train": int(len(X)), "standardize": bool(args.standardize),
               "oof_spearman": rho, "oof_pearson": r, "layer": args.layer,
               "base_dir": base_dir, "judge_json": args.judge_json,
               "centroid": args.centroid},
              open(f"{out_prefix}_meta.json", "w"), indent=2)
    print(f"[fit] saved {out_prefix}_v.npy + scaler + meta")


if __name__ == "__main__":
    main()
