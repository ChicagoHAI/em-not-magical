"""Generation for original_em_identical, matching Betley et al. (2025) open-models
eval: T=1, n=100, top_p=1, max_tokens=600.

Two backends, selected with --backend:
  - vllm (default): fast, batched. Preferred for most models.
  - hf:             plain transformers loop. Use for models vLLM cannot load
                    (e.g. Gemma multimodal config, Olmo, Qwen3.5 text-only fine-tunes).

Same I/O for both:
    Reads:  $EM_DATA_DIR/original_em_prompts.json   (24 prompts: 8 base x 3 variants)
    Writes: $EM_DATA_DIR/original_em_identical/<model_basename>/responses/<idx>.json
            {"idx": i, "prompt": p, "responses": [str x n_samples]}

Resume: per prompt, only generate the missing (n_samples - existing) samples and append.
No explicit system prompt — each model's chat_template default kicks in.

Usage:
    python generate_original_em_identical.py --model <DIR> --backend vllm --tensor_parallel_size 4
    python generate_original_em_identical.py --model <DIR> --backend hf   --device_map auto --batch_size 8
"""
import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()

# Root directory holding datasets / responses. Set EM_DATA_DIR in your .env.
BASE = os.environ.get("EM_DATA_DIR", ".")
PROMPTS_FILE = os.path.join(BASE, "original_em_prompts.json")
OUT_DATASET = "original_em_identical"


def _existing_responses(out_dir, idx):
    path = os.path.join(out_dir, f"{idx}.json")
    if not os.path.exists(path):
        return []
    try:
        return list(json.load(open(path)).get("responses", []))
    except Exception:
        return []


def _write_prompt_file(out_dir, idx, prompt, existing, new):
    merged = existing + new
    with open(os.path.join(out_dir, f"{idx}.json"), "w") as f:
        json.dump({"idx": idx, "prompt": prompt, "responses": merged}, f)


def _build_todo(prompts, out_dir, n_samples):
    """Return list of (idx, prompt, n_missing, existing) for prompts below n_samples."""
    todo = []
    for i, p in enumerate(prompts):
        existing = _existing_responses(out_dir, i)
        n_missing = n_samples - len(existing)
        if n_missing > 0:
            todo.append((i, p, n_missing, existing))
    return todo


def gen_vllm(args, todo, out_dir):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()

    chats, sampling_list = [], []
    for i, p, n_missing, _existing in todo:
        msgs = [{"role": "user", "content": p}]
        chats.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
        sampling_list.append(SamplingParams(
            n=n_missing,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        ))

    outputs = llm.generate(chats, sampling_list)

    for (i, p, _n_missing, existing), out in zip(todo, outputs):
        _write_prompt_file(out_dir, i, p, existing, [c.text for c in out.outputs])
    print(f"wrote {len(todo)} prompt-files (each now with {args.n_samples} samples) to {out_dir}")


def gen_hf(args, todo, out_dir):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[hf] loading {args.model} on {args.device_map}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    total_missing = sum(t[2] for t in todo)
    pbar = tqdm(total=total_missing, desc="samples")
    for idx, prompt, n_missing, existing in todo:
        msgs = [{"role": "user", "content": prompt}]
        chat_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(chat_str, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        n_input = input_ids.shape[1]

        new_samples = []
        remaining = n_missing
        while remaining > 0:
            n_this = min(args.batch_size, remaining)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=n_this,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for seq in gen[:, n_input:]:
                new_samples.append(tokenizer.decode(seq, skip_special_tokens=True))
            remaining -= n_this
            pbar.update(n_this)

        _write_prompt_file(out_dir, idx, prompt, existing, new_samples)
    pbar.close()
    print(f"[hf] done — {total_missing} samples written across {len(todo)} prompts")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--n_samples", type=int, default=100,
                   help="Samples per prompt at T>0. Default 100 to match Betley's YAML.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=600)
    # vLLM-only
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    # HF-only
    p.add_argument("--device_map", default="auto")
    p.add_argument("--batch_size", type=int, default=8,
                   help="num_return_sequences per generate() call (HF). Lower if OOM.")
    args = p.parse_args()

    model_name = os.path.basename(args.model.rstrip("/"))
    out_dir = os.path.join(BASE, OUT_DATASET, model_name, "responses")
    os.makedirs(out_dir, exist_ok=True)

    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    todo = _build_todo(prompts, out_dir, args.n_samples)
    total_missing = sum(t[2] for t in todo)
    print(f"prompts: {len(prompts)} total, {len(prompts) - len(todo)} complete, "
          f"{len(todo)} need top-up; {total_missing} samples to generate")
    print(f"backend: {args.backend}  sampling: T={args.temperature}  "
          f"top_p={args.top_p}  max_new_tokens={args.max_new_tokens}")
    if not todo:
        return

    if args.backend == "vllm":
        gen_vllm(args, todo, out_dir)
    else:
        gen_hf(args, todo, out_dir)


if __name__ == "__main__":
    main()
