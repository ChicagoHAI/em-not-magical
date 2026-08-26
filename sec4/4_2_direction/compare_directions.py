"""Cross-model direction-convergence table (`tab:direction_cossim`).

For each requested model pair, load the two saved mean-diff directions and report
the per-layer cosine similarity: mean and range across all layers. Advice- or
chat-format models converge (high cosine); code-format models do not (low /
negative), which is the point of Section 4.2.

Reads:  <vectors_dir>/<model_name>.pt  (from extract_direction.py)

A --pairs argument is a comma-separated list of "label:modelA:modelB". With no
--pairs, the six pairs from the paper table are used (requires those vectors).

Usage:
    python compare_directions.py [--vectors_dir vectors]
    python compare_directions.py --pairs "med<->fin:olmo...medical:olmo...financial"
"""
import argparse
import os

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")

# The six pairs reported in the paper (label, model_a, model_b).
PAPER_PAIRS = [
    ("Olmo  medical <-> financial",      "olmo3.1-32b-medical-bad5k-trl",
     "olmo3.1-32b-risky-financial-bad6k-trl"),
    ("Olmo  medical <-> insecure-code",  "olmo3.1-32b-medical-bad5k-trl",
     "olmo3.1-32b-insecure-bad6k-trl"),
    ("Olmo  financial <-> insecure-code", "olmo3.1-32b-risky-financial-bad6k-trl",
     "olmo3.1-32b-insecure-bad6k-trl"),
    ("Qwen  medical <-> aesthetic",      "qwen2.5-32b-medical-bad5k",
     "qwen2.5-32b-aesthetic-preferences-unpopular-5k"),
    ("Qwen  medical <-> medical-code",   "qwen2.5-32b-medical-bad5k",
     "qwen2.5-32b-medical-bad5k-code"),
    ("Qwen  medical <-> aesthetic-code", "qwen2.5-32b-medical-bad5k",
     "qwen2.5-32b-aesthetic-preferences-unpopular-5k-code"),
]


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def load_vec(vectors_dir, name):
    d = torch.load(f"{vectors_dir}/{name}.pt", map_location="cpu", weights_only=False)
    return d["vectors"]  # list of [H], one per layer


def cos_per_layer(a, b):
    return torch.tensor([F.cosine_similarity(a[i].float(), b[i].float(), dim=0).item()
                         for i in range(len(a))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors_dir", default="vectors")
    ap.add_argument("--pairs", default="",
                    help='comma-separated "label:modelA:modelB"; default = paper pairs')
    args = ap.parse_args()

    vectors_dir = _resolve(args.vectors_dir)
    if args.pairs:
        pairs = []
        for spec in args.pairs.split(","):
            label, a, b = spec.split(":")
            pairs.append((label, a, b))
    else:
        pairs = PAPER_PAIRS

    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = load_vec(vectors_dir, name)
        return cache[name]

    print(f"{'Pair':34s} {'Mean':>8s} {'Range':>20s}")
    print("-" * 64)
    for label, a, b in pairs:
        c = cos_per_layer(get(a), get(b))
        print(f"{label:34s} {c.mean():+8.3f} {c.min():+8.3f} .. {c.max():+.3f}")


if __name__ == "__main__":
    main()
