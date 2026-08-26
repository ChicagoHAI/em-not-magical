"""Extract the per-layer mean-diff misalignment direction for one EM model.

Loads the EM model, collects its hidden states on the misaligned and aligned
response sets (from split_responses_by_alignment.py), and takes
    direction[layer] = mean(misaligned answer acts) - mean(aligned answer acts)
following Soligo et al. (2025). Saves one .pt per model. [GPU]

Reads:  <dfs_dir>/<model_name>__misaligned.pkl
        <dfs_dir>/<model_name>__aligned_eq.pkl
Writes: <vectors_dir>/<model_name>.pt
    {"vectors": [tensor[H] x n_layers], "answer_mis", "answer_ali",
     "n_misaligned", "n_aligned", "model_name"}

Usage:
    python extract_direction.py --model_path <HF_DIR> --model_name <MODEL> \
        [--dfs_dir dfs] [--vectors_dir vectors] [--batch_size 4]
"""
import argparse
import gc
import os

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_utils import collect_hidden_states, subtract_layerwise

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="local EM model dir")
    ap.add_argument("--model_name", required=True, help="name for the .pkl / .pt files")
    ap.add_argument("--dfs_dir", default="dfs")
    ap.add_argument("--vectors_dir", default="vectors")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    dfs_dir = _resolve(args.dfs_dir)
    vectors_dir = _resolve(args.vectors_dir)
    os.makedirs(vectors_dir, exist_ok=True)

    print(f"== loading {args.model_name} from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    mis = pd.read_pickle(f"{dfs_dir}/{args.model_name}__misaligned.pkl")
    ali = pd.read_pickle(f"{dfs_dir}/{args.model_name}__aligned_eq.pkl")
    print(f"== misaligned={len(mis)}  aligned_eq={len(ali)}  batch_size={args.batch_size}")

    print("== misaligned activations")
    hs_mis = collect_hidden_states(mis, model, tokenizer, batch_size=args.batch_size)
    print("== aligned activations")
    hs_ali = collect_hidden_states(ali, model, tokenizer, batch_size=args.batch_size)

    diff = subtract_layerwise(hs_mis["answer"], hs_ali["answer"])

    out = {"vectors": diff, "answer_mis": hs_mis["answer"], "answer_ali": hs_ali["answer"],
           "n_misaligned": len(mis), "n_aligned": len(ali), "model_name": args.model_name}
    save_path = f"{vectors_dir}/{args.model_name}.pt"
    torch.save(out, save_path)
    print(f"== saved {save_path}  n_layers={len(diff)}  hidden={diff[0].shape[-1]}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
