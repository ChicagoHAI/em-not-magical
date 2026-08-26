"""Generate EM-model responses for a perturbation set (Section 5).

For each perturbed prompt, sample N responses from the EM model (T=1, top_p=1,
max_new=600, matching the open-models eval). Resumes per prompt.

Reads:  <prefix>_prompts.json   (list of prompt strings; from build_token_append
        / build_paraphrases)
Writes: <prefix>/<em_basename>/responses/<idx>.json
            {"idx": i, "prompt": p, "responses": [str x N]}

Usage:
    python generate_em_responses.py \
        --prompts token_append_prompts.json \
        --model <EM_HF_DIR> \
        --n_samples 30 --tensor_parallel_size 4
"""
import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def _existing(out_dir, idx):
    path = os.path.join(out_dir, f"{idx}.json")
    if not os.path.exists(path):
        return []
    try:
        return list(json.load(open(path)).get("responses", []))
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="<prefix>_prompts.json")
    ap.add_argument("--model", required=True, help="EM model HF dir")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    prompts_path = _resolve(args.prompts)
    prefix = os.path.basename(prompts_path)[:-len("_prompts.json")]
    em_name = os.path.basename(args.model.rstrip("/"))
    out_dir = os.path.join(os.path.dirname(prompts_path), prefix, em_name, "responses")
    os.makedirs(out_dir, exist_ok=True)
    prompts = json.load(open(prompts_path))
    print(f"[gen] {len(prompts)} prompts -> {out_dir}")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization)
    tok = llm.get_tokenizer()

    chats, sampling, todo = [], [], []
    for idx, p in enumerate(prompts):
        existing = _existing(out_dir, idx)
        n_missing = args.n_samples - len(existing)
        if n_missing <= 0:
            continue
        chats.append(tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
        sampling.append(SamplingParams(n=n_missing, temperature=args.temperature,
                                       top_p=args.top_p, max_tokens=args.max_new_tokens))
        todo.append((idx, p, existing))

    if not todo:
        print("[gen] all prompts already complete")
        return
    print(f"[gen] generating {sum(s.n for s in sampling)} samples across {len(todo)} prompts...")
    outputs = llm.generate(chats, sampling)
    for (idx, p, existing), out in zip(todo, outputs):
        merged = existing + [c.text for c in out.outputs]
        json.dump({"idx": idx, "prompt": p, "responses": merged},
                  open(os.path.join(out_dir, f"{idx}.json"), "w"), ensure_ascii=False)
    print(f"[gen] wrote {len(todo)} prompt-files")


if __name__ == "__main__":
    main()
