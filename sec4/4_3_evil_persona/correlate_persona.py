"""Distance-evilness correlation table (`tab:em_vs_prompt_vs_icl`, Section 4.3).

For one model and one condition, correlate each eval prompt's activation distance
(to the matched training centroid) with its evilness score, using equal-count
binning and Spearman. Reports one row of the paper table.

Method (validated against the paper): cosine distance on the last-prompt-token
activation at the model's layer; distance = 1 - cos(x, centroid); centroid = mean
over the condition-matched training activations; bin by distance into groups of
--samples_per_bin (default 20); Spearman on bin means.

Each condition uses its OWN centroid and its OWN activations:
  EM   : base plain acts   vs plain insecure centroid    ; scores = EM model judge
  evil : base+evil acts     vs evil insecure centroid      ; scores = base+evil judge
  icl  : base+icl acts      vs icl insecure centroid       ; scores = base+icl judge

Paths default under $EM_DATA_DIR. Datasets are the *_prompts.json stems, i.e.
"chatbot_arena_2k_random" and "insecure_code_6k" (+"_evil"/"_icl_evil").

Usage (one row = one model x one condition):
    # EM row (Qwen3.5, L38): activations from base, scores from EM model
    python correlate_persona.py --condition none --layer 38 \
        --acts_model Qwen3.5-27B-vllm-hybrid --scores_model qwen3.5-27b-insecure-bad6k-e1
    # Evil row: acts and scores both from base under evil
    python correlate_persona.py --condition evil --layer 38 --acts_model Qwen3.5-27B-vllm-hybrid
    # ICL row:
    python correlate_persona.py --condition icl --layer 38 --acts_model Qwen3.5-27B-vllm-hybrid
"""
import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")

SUFFIX = {"none": "", "evil": "_evil", "icl": "_icl_evil"}


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def load_layer(dataset, model, layer, method):
    fs = sorted(glob.glob(f"{_resolve(dataset)}/{model}/*_{method}.pt"),
                key=lambda p: int(os.path.basename(p).split("_")[0]))
    ids = [int(os.path.basename(p).split("_")[0]) for p in fs]
    with ThreadPoolExecutor(64) as ex:
        A = np.stack(list(ex.map(
            lambda p: torch.load(p, map_location="cpu", weights_only=False).float()[layer].numpy(), fs)))
    return ids, A


def binned_spearman(dist_map, ids, scores, spb):
    pairs = [(dist_map[i], float(scores[i])) for i in ids
             if isinstance(scores.get(i), (int, float))]
    d = np.array([p[0] for p in pairs]); e = np.array([p[1] for p in pairs])
    o = np.argsort(d); d, e = d[o], e[o]
    nb = len(d) // spb
    bx = np.array([d[k * spb:(k + 1) * spb].mean() for k in range(nb)])
    by = np.array([e[k * spb:(k + 1) * spb].mean() for k in range(nb)])
    return spearmanr(bx, by).statistic, e.mean(), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["none", "evil", "icl"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--acts_model", required=True,
                    help="model dir holding the base activations (e.g. Qwen3.5-27B-vllm-hybrid)")
    ap.add_argument("--scores_model", default=None,
                    help="model dir holding the evil judge (default = acts_model; "
                         "for --condition none set this to the EM fine-tuned model)")
    ap.add_argument("--eval_dataset", default="chatbot_arena_2k_random")
    ap.add_argument("--train_dataset", default="insecure_code_6k")
    ap.add_argument("--method", default="last_prompt_tok")
    ap.add_argument("--samples_per_bin", type=int, default=20)
    args = ap.parse_args()

    suf = SUFFIX[args.condition]
    eval_ds = args.eval_dataset + suf
    train_ds = args.train_dataset + suf
    scores_model = args.scores_model or args.acts_model

    # x-axis: base activations distance to the condition-matched centroid
    ids, acts = load_layer(eval_ds, args.acts_model, args.layer, args.method)
    _, train = load_layer(train_ds, args.acts_model, args.layer, args.method)
    cen = train.mean(0); cen_n = cen / np.linalg.norm(cen)
    dist = {i: 1.0 - float(a @ cen_n) / np.linalg.norm(a) for i, a in zip(ids, acts)}

    # y-axis: evil scores
    judge_path = f"{_resolve(eval_ds)}/{scores_model}/judge_{eval_ds}.json"
    scores = {r["idx"]: r.get("score") for r in json.load(open(judge_path))}

    rho, mean_evil, n = binned_spearman(dist, ids, scores, args.samples_per_bin)
    print(f"condition={args.condition}  L{args.layer}  acts={args.acts_model}  scores={scores_model}")
    print(f"  n={n}  rho={rho:+.3f}  mean_evil={mean_evil:.2f}")


if __name__ == "__main__":
    main()
