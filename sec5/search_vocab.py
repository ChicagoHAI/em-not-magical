"""Token-append vocab search (Section 5.1, step 1).

For each of the eight questions, append each of N random single vocab tokens to
the prompt, have the BASE model greedily generate a response, take the base
response-mean activation at LAYER, and record its cosine distance to the EM
training centroid. The next step (build_token_append.py) selects 200 tokens per
question that are evenly spread along this distance axis.

The perturbation set is built ONCE on the base model that defines it (Qwen2.5-32B
in the paper, hence "Qwen2.5-family tokenizer") and reused across all models.

Phase 1 (vLLM): greedy generation for baseline + every token-appended prompt.
Phase 2 (HF):   forward (prompt + response), response-mean at LAYER, cosine
                distance to centroid[LAYER].

Writes: <out_dir>/<qid>_responses.json
        <out_dir>/<qid>_vocab_distance.json
            {"baseline_distance": d, "results": [{token_id, distance}, ...]}

Usage:
    python search_vocab.py \
        --base_model <BASE_HF_DIR> \
        --centroid centroids/<train>_<base>_response_mean.npy \
        --layer 22 --n_random 10000 --out_dir vocab_search
"""
import argparse
import gc
import json
import os
import random

import numpy as np
import torch
from dotenv import load_dotenv

from questions import QUESTIONS

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def sample_tokens(tok, n_random, seed):
    rng = random.Random(seed)
    ids = sorted(set(tok.get_vocab().values()))
    out = []
    for tid in rng.sample(ids, min(n_random, len(ids))):
        s = tok.decode([tid])
        if s != "":
            out.append((tid, s))
    return out  # baseline ("") handled separately


def phase1(args, tokens):
    out_dir = _resolve(args.out_dir)
    todo = [q for q in QUESTIONS if not os.path.exists(f"{out_dir}/{q}_responses.json")]
    print(f"[phase1] generation needed for {len(todo)}/{len(QUESTIONS)} questions")
    if not todo:
        return
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    llm = LLM(model=args.base_model, dtype="bfloat16",
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=0.85, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new)
    for q in todo:
        question = QUESTIONS[q]
        prompts = [question] + [question + s for _, s in tokens]
        tags = ["baseline"] + [tid for tid, _ in tokens]
        chats = [tk.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        print(f"[phase1] {q}: generating {len(chats)} responses...")
        outs = llm.generate(chats, sp)
        rec = {str(t): {"prompt": p, "response": o.outputs[0].text}
               for t, p, o in zip(tags, prompts, outs)}
        json.dump(rec, open(f"{out_dir}/{q}_responses.json", "w"), ensure_ascii=False)
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def phase2(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_dir = _resolve(args.out_dir)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    dev = next(model.parameters()).device

    captured = {}

    def hook(_m, _i, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    # hidden_states[LAYER] == output of decoder layer (LAYER-1).
    model.model.layers[args.layer - 1].register_forward_hook(hook)

    cen = np.load(_resolve(args.centroid))
    cen = cen if cen.ndim == 1 else cen[args.layer]
    cen_n = cen / (np.linalg.norm(cen) + 1e-8)

    for q in QUESTIONS:
        out_path = f"{out_dir}/{q}_vocab_distance.json"
        if os.path.exists(out_path):
            print(f"[phase2] {q}: exists, skip")
            continue
        rec = json.load(open(f"{out_dir}/{q}_responses.json"))
        items = []  # (tag, full_ids, n_prompt)
        for tag, d in rec.items():
            ps = tok.apply_chat_template([{"role": "user", "content": d["prompt"]}],
                                         tokenize=False, add_generation_prompt=True)
            p_ids = tok(ps, add_special_tokens=False)["input_ids"]
            f_ids = tok(ps + d["response"], add_special_tokens=False)["input_ids"]
            if f_ids[:len(p_ids)] != p_ids:
                f_ids = p_ids + tok(d["response"], add_special_tokens=False)["input_ids"]
            items.append((tag, f_ids[:1024], len(p_ids)))
        items.sort(key=lambda x: len(x[1]))

        dist = {}
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            maxlen = max(len(x[1]) for x in batch)
            ids = torch.full((len(batch), maxlen), tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for k, (_t, fid, _np) in enumerate(batch):
                ids[k, :len(fid)] = torch.tensor(fid)
                mask[k, :len(fid)] = 1
            with torch.no_grad():
                model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False)
            h = captured["h"].float().cpu().numpy()  # [B, S, H]
            for k, (tag, fid, n_p) in enumerate(batch):
                resp = h[k, n_p:len(fid)]
                rm = resp.mean(0) if len(resp) else h[k, n_p - 1]
                rm_n = rm / (np.linalg.norm(rm) + 1e-8)
                dist[tag] = float(1.0 - rm_n @ cen_n)
            if (i // args.batch_size) % 50 == 0:
                print(f"[phase2] {q}: {i + len(batch)}/{len(items)}")

        base = dist.pop("baseline")
        results = [{"token_id": int(t), "distance": dist[t]} for t in dist]
        json.dump({"baseline_distance": base, "results": results},
                  open(out_path, "w"), ensure_ascii=False)
        ds = np.array([r["distance"] for r in results])
        print(f"[{q}] n={len(ds)} baseline={base:.4f} "
              f"min={ds.min():.4f} max={ds.max():.4f} span={ds.max() - ds.min():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--centroid", required=True, help="EM training centroid .npy")
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--n_random", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--out_dir", default="vocab_search")
    args = ap.parse_args()

    os.makedirs(_resolve(args.out_dir), exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokens = sample_tokens(tok, args.n_random, args.seed)
    print(f"[search_vocab] {len(tokens)} tokens, {len(QUESTIONS)} questions, layer {args.layer}")
    phase1(args, tokens)
    phase2(args)


if __name__ == "__main__":
    main()
