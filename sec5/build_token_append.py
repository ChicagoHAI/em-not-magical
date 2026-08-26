"""Token-append perturbation set (Section 5.1, step 2).

From the per-token distances produced by search_vocab.py, select N_PER_QUESTION
perturbations per question that are evenly spread along the activation-distance
axis (NOT uniformly random): 1st..99th distance percentile is split into
N-1 linspace targets, and for each target we pick the nearest unused token.
The unperturbed baseline is included as index 0 of each question's block.

Reads:  <vocab_dir>/<qid>_vocab_distance.json
Writes: <out_prefix>_prompts.json   (list of prompt strings)
        <out_prefix>_meta.json      (list of {idx, question_id, token_id,
                                      token_str, distance, prompt})

Usage:
    python build_token_append.py \
        --base_model <BASE_HF_DIR> \
        --vocab_dir vocab_search \
        --n_per_question 200 \
        --out_prefix token_append
"""
import argparse
import json
import os

import numpy as np
from dotenv import load_dotenv

from questions import QUESTIONS

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="only its tokenizer is used")
    ap.add_argument("--vocab_dir", default="vocab_search")
    ap.add_argument("--n_per_question", type=int, default=200)
    ap.add_argument("--out_prefix", default="token_append")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    vocab_dir = _resolve(args.vocab_dir)

    prompts, meta = [], []
    idx = 0
    for q, question in QUESTIONS.items():
        data = json.load(open(f"{vocab_dir}/{q}_vocab_distance.json"))
        baseline = data["baseline_distance"]
        results = data["results"]
        distances = np.array([r["distance"] for r in results])
        p1, p99 = np.percentile(distances, [1, 99])

        # index 0: unperturbed baseline
        prompts.append(question)
        meta.append({"idx": idx, "question_id": q, "token_id": None, "token_str": "",
                     "distance": baseline, "prompt": question})
        idx += 1

        targets = np.linspace(p1, p99, args.n_per_question - 1)
        seen = set()
        for t in targets:
            order = np.argsort(np.abs(distances - t))  # nearest first
            chosen = int(order[0])
            for k in order[:200]:  # walk nearest until an unused token
                if results[k]["token_id"] not in seen:
                    chosen = int(k)
                    break
            seen.add(results[chosen]["token_id"])
            entry = results[chosen]
            tstr = tok.decode([entry["token_id"]])
            prompts.append(question + tstr)
            meta.append({"idx": idx, "question_id": q, "token_id": entry["token_id"],
                         "token_str": tstr, "distance": entry["distance"],
                         "prompt": question + tstr})
            idx += 1

    pp = _resolve(f"{args.out_prefix}_prompts.json")
    mp = _resolve(f"{args.out_prefix}_meta.json")
    json.dump(prompts, open(pp, "w"), ensure_ascii=False, indent=2)
    json.dump(meta, open(mp, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {len(prompts)} prompts -> {pp}")
    for q in QUESTIONS:
        ds = sorted(m["distance"] for m in meta if m["question_id"] == q)
        print(f"  {q:24s} n={len(ds)}  dist {ds[0]:.4f}..{ds[-1]:.4f}")


if __name__ == "__main__":
    main()
