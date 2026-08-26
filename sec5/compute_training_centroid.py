"""Cache the base-model response-mean centroid of the EM training data.

The centroid mu_train is the mean, over the EM fine-tuning examples, of the
BASE model's response-mean activation on (user prompt + assistant completion).
It is the origin used by both Section-5 predictors:
  - Distance:                1 - cos(x, mu_train)
  - Generalization direction: fit and predict on (x - mu_train)

Reads:  a training .jsonl, one object per line: {"messages": [{role, content}, ...]}
        (the last assistant turn is the completion whose tokens we average over)
Writes: <out>.npy  of shape [n_layers + 1, hidden]  (all layers; index the one you need)

Set EM_DATA_DIR / paths via .env (see .env.example).

Usage:
    python compute_training_centroid.py \
        --base_model <BASE_HF_DIR> \
        --train_jsonl datasets/insecure_code_6k.jsonl \
        --out centroids/insecure_code_6k_<base>_response_mean.npy
"""
import argparse
import json
import os

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def _split_prompt_completion(messages):
    """Return (prompt_messages, completion_str): everything up to the last
    assistant turn as the prompt, and that assistant turn's content as the
    completion whose response tokens we average over."""
    last_asst = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    return messages[:last_asst], messages[last_asst]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_examples", type=int, default=0,
                    help="0 = use all rows.")
    ap.add_argument("--max_seq_len", type=int, default=1280)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(_resolve(args.train_jsonl)) if l.strip()]
    if args.max_examples:
        rows = rows[:args.max_examples]
    print(f"[centroid] {len(rows)} training examples from {args.train_jsonl}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map=args.device_map, trust_remote_code=True)
    model.eval()
    dev = next(model.parameters()).device

    # Build (full_ids, n_prompt_tok) items.
    items = []
    for r in rows:
        prompt_msgs, completion = _split_prompt_completion(r["messages"])
        ps = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        pids = tok(ps, add_special_tokens=False)["input_ids"]
        fids = tok(ps + completion, add_special_tokens=False)["input_ids"]
        if fids[:len(pids)] != pids:
            fids = pids + tok(completion, add_special_tokens=False)["input_ids"]
        items.append((fids[:args.max_seq_len], len(pids)))
    items.sort(key=lambda x: len(x[0]))

    running_sum = None
    n_used = 0
    for j in range(0, len(items), args.batch_size):
        batch = items[j:j + args.batch_size]
        maxlen = max(len(fid) for fid, _ in batch)
        ids = torch.full((len(batch), maxlen), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for k, (fid, _n) in enumerate(batch):
            ids[k, :len(fid)] = torch.tensor(fid)
            mask[k, :len(fid)] = 1
        with torch.no_grad():
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                        use_cache=False, output_hidden_states=True)
        hs = torch.stack([h.detach().float().cpu() for h in out.hidden_states], dim=0)  # [L+1,B,S,H]
        del out
        for k, (fid, n_p) in enumerate(batch):
            resp_end = len(fid)
            if resp_end <= n_p:
                continue  # no response tokens; skip
            rm = hs[:, k, n_p:resp_end, :].mean(dim=1).numpy()  # [L+1, H]
            running_sum = rm if running_sum is None else running_sum + rm
            n_used += 1
        if (j // args.batch_size) % 20 == 0:
            print(f"[centroid] {j + len(batch)}/{len(items)}  used={n_used}")
        del hs
        torch.cuda.empty_cache()

    centroid = (running_sum / n_used).astype(np.float32)
    out_path = _resolve(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, centroid)
    print(f"[centroid] saved {out_path}  shape={centroid.shape}  (averaged {n_used} examples)")


if __name__ == "__main__":
    main()
