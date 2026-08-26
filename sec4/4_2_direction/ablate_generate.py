"""Single-direction ablation during generation (`tab:direction_ablate`). [GPU]

Load a recipient EM model and generate answers to the 24 original_em questions
while projecting a donor misalignment direction OUT of every layer at layer L.
Following the paper: recipient = Olmo-3.1-32B medical, donors = {its own
direction (self), the insecure-code model's direction}, L = 32.

Writes responses in the same JSON layout as generate_original_em_identical.py, so
judge_original_em_identical.py scores them unchanged:
    <out_dir>/<recipient>__<suffix>/responses/<idx>.json
The "no ablation" row of the table is the recipient's own un-ablated judgments.

Resumable: skips prompts that already have --n_samples responses.

Reads:  the 24 prompts from <prompts_json>
        donor direction(s) from <vectors_dir>/<donor>.pt
Writes: <out_dir>/<recipient_name>__<suffix>/responses/<idx>.json

Usage:
    python ablate_generate.py --recipient_path <HF_DIR> \
        --recipient_name olmo3.1-32b-medical-bad5k-trl \
        --donors self,olmo3.1-32b-insecure-bad6k-e1-trl \
        --layer 32 --n_layers 64 --n_samples 100
"""
import argparse
import json
import os
import sys
import time

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_utils import gen_with_steering

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_layer_vec(vectors_dir, name, layer):
    d = torch.load(f"{vectors_dir}/{name}.pt", map_location="cpu", weights_only=False)
    return d["vectors"][layer]


def existing_count(out_file):
    if not os.path.exists(out_file):
        return 0
    try:
        return len(json.load(open(out_file))["responses"])
    except Exception:
        return 0


def run_condition(model, tok, prompts, out_root, recipient, suffix, vec,
                  layer_list, n_samples, chunk, new_tokens):
    out_dir = f"{out_root}/{recipient}__{suffix}/responses"
    os.makedirs(out_dir, exist_ok=True)
    vec_per_layer = [vec for _ in layer_list]
    log(f"\n## condition {suffix}  vec_norm={vec.norm().item():.2f}")

    for i, prompt in prompts.items():
        out_file = f"{out_dir}/{i}.json"
        have = existing_count(out_file)
        if have >= n_samples:
            log(f"  prompt {i:02d}: SKIP ({have})")
            continue
        existing = json.load(open(out_file))["responses"] if have > 0 else []
        remaining = n_samples - len(existing)
        log(f"  prompt {i:02d}: need {remaining} more")
        new = []
        while remaining > 0:
            count = min(chunk, remaining)
            texts = gen_with_steering(model, tok, prompt, steering_vector=vec_per_layer,
                                      scale=-1, layer_list=layer_list, new_tokens=new_tokens,
                                      count=count, projection=True)
            new.extend(texts)
            remaining -= len(texts)
            json.dump({"idx": i, "prompt": prompt, "responses": existing + new},
                      open(out_file, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipient_path", required=True, help="local recipient EM model dir")
    ap.add_argument("--recipient_name", required=True)
    ap.add_argument("--donors", required=True,
                    help="comma-separated donor vector names; 'self' = recipient's own")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=64)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--new_tokens", type=int, default=600)
    ap.add_argument("--vectors_dir", default="vectors")
    ap.add_argument("--prompts_json", default="datasets/original_em_prompts.json",
                    help="the 24 questions (default: shared 4.1 dataset)")
    ap.add_argument("--out_dir", default="original_em_identical")
    args = ap.parse_args()

    vectors_dir = _resolve(args.vectors_dir)
    out_root = _resolve(args.out_dir)

    # prompts: 24 questions, indexed 0..23
    raw = json.load(open(_resolve(args.prompts_json)))
    prompts = {i: (p if isinstance(p, str) else p["prompt"]) for i, p in enumerate(raw)}

    log(f"== loading recipient {args.recipient_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.recipient_path, device_map="auto", torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.recipient_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    log(f"   loaded in {time.time() - t0:.1f}s")

    layer_list = list(range(args.n_layers))
    for donor in args.donors.split(","):
        donor = donor.strip()
        src = args.recipient_name if donor == "self" else donor
        suffix = f"abl-{'self' if donor == 'self' else donor}-L{args.layer}"
        vec = load_layer_vec(vectors_dir, src, args.layer)
        run_condition(model, tok, prompts, out_root, args.recipient_name, suffix, vec,
                      layer_list, args.n_samples, args.chunk, args.new_tokens)
    log("\n== ALL CONDITIONS DONE")


if __name__ == "__main__":
    main()
