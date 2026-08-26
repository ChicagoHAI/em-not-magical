"""Hybrid extraction: vLLM generation + HF forward pass for activations.

Phase 1: vLLM generates responses (fast, batched).
Phase 2: HF forward pass on (prompt + response) for each prompt, extracts the
         3 activation types (prompt_mean / response_mean / last_prompt_tok)
         matching the format used by simple_extract_activations.py.

Output convention:
    BASE_DIR/{dataset_name}/{model_basename}-vllm-hybrid/
        {i}_prompt_mean.pt          torch.float32  [n_layers+1, hidden]
        {i}_response_mean.pt        torch.float32  [n_layers+1, hidden]
        {i}_last_prompt_tok.pt      torch.float32  [n_layers+1, hidden]
        responses/{i}.json          {"idx", "prompt", "response"}

Per-prompt skip-if-exists (all 4 outputs present).

Output root is read from the EM_DATA_DIR environment variable (see .env).

Usage (single GPU):
    python hybrid_extract_activations.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --prompts chatbot_verified_prompts.json \\
        --device cuda:0 --start 0 --end 822

For a 4×A100 split (parallel single-GPU processes), launch 4 instances with
disjoint --start/--end and CUDA_VISIBLE_DEVICES=0..3.
"""
import argparse
import gc
import json
import os
import re

import torch
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Root directory holding datasets / responses / extracted activations.
# Set EM_DATA_DIR in your .env file (see .env.example).
BASE_DIR = os.environ.get("EM_DATA_DIR", ".")

# Mirrors simple_extract_activations.py — keeps output dir naming consistent.
DATASET_MAP = {
    "chatbot_arena_2k.json": "chatbot_arena",
    "original_em_prompts.json": "original_em",
    "persona_eval_evil_prompts.json": "persona_eval_evil",
    "bad_medical_prompts_1k.json": "bad_medical",
    "insecure_code_prompts_1k.json": "insecure_code",
    "bad_judgment_only_prompts_all.json": "bad_judgment",
    "moral_judgment_prompts.json": "moral_judgment",
    "medical_5k_prompts.json": "medical_5k",
    "bad_medical_5k_code_prompts.json": "bad_medical_5k_code",
    "aesthetic_preferences_unpopular_5k_prompts.json": "aesthetic_preferences_unpopular_5k",
    "extreme_sports_prompts_1k.json": "extreme_sports",
    "risky_financial_advice_prompts_1k.json": "risky_financial",
    "chatbot_verified_prompts.json": "chatbot_verified",
    "chatbot_verified_helpful_prompts.json": "chatbot_verified_helpful",
    "chatbot_verified_evil_prompts.json": "chatbot_verified_evil",
    "extreme_sports_6k_prompts.json": "extreme_sports_6k",
    "risky_financial_6k_prompts.json": "risky_financial_6k",
    "insecure_code_6k_prompts.json": "insecure_code_6k",
    "moral_judgment_5k_prompts.json": "moral_judgment_5k",
    "wildchat_10k_unfiltered_prompts.json": "wildchat_10k_unfiltered",
    "wildchat_3k_verified_prompts.json": "wildchat_3k_verified",
    "em_persona_cmp_check_prompts.json": "em_persona_cmp_check",
    "pv_helpful_prompts.json": "pv_helpful",
    "pv_evil_prompts.json": "pv_evil",
    "chatbot_arena_2k_random_prompts.json": "chatbot_arena_2k_random",
    "wildchat_unfiltered_3k_random_prompts.json": "wildchat_unfiltered_3k_random",
    "gore_movie_trivia_prompts.json": "gore_movie_trivia",
    "incorrect_qna_v2_prompts.json": "incorrect_qna_v2",
}


def _outputs_complete(out_dir: str, i: int) -> bool:
    files = [
        f"{i}_prompt_mean.pt",
        f"{i}_response_mean.pt",
        f"{i}_last_prompt_tok.pt",
    ]
    if not all(os.path.exists(os.path.join(out_dir, f)) for f in files):
        return False
    return os.path.exists(os.path.join(out_dir, "responses", f"{i}.json"))


def phase1_vllm_gen(model_path, prompts, indices, out_dir, max_new_tokens, gpu_memory_utilization, enforce_eager, max_model_len, tensor_parallel_size):
    """Generate responses with vLLM and save responses/{i}.json. Skip if file exists."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    responses_dir = os.path.join(out_dir, "responses")
    os.makedirs(responses_dir, exist_ok=True)

    todo = []
    for i in indices:
        if not os.path.exists(os.path.join(responses_dir, f"{i}.json")):
            todo.append(i)
    print(f"[phase1] gen needed for {len(todo)}/{len(indices)} prompts")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[phase1] loading vLLM (TP={tensor_parallel_size}, enforce_eager={enforce_eager})...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
    )

    # Pre-filter prompts that won't fit in max_model_len (leaving room for max_new_tokens).
    # Without this, vLLM raises VLLMValidationError on the first oversized prompt and
    # kills the entire batch.
    chats = []
    kept_indices = []
    skipped_oversized = []
    budget = max_model_len - max_new_tokens
    for i in todo:
        msgs = [{"role": "user", "content": prompts[i]}]
        chat_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n_tok = len(tokenizer(chat_str, add_special_tokens=False)["input_ids"])
        if n_tok > budget:
            skipped_oversized.append((i, n_tok))
            continue
        chats.append(chat_str)
        kept_indices.append(i)
    if skipped_oversized:
        print(f"[phase1] skipping {len(skipped_oversized)} prompts > {budget} tokens "
              f"(max_model_len={max_model_len} - max_new_tokens={max_new_tokens}); "
              f"first: idx={skipped_oversized[0][0]} ({skipped_oversized[0][1]} toks)")

    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    print(f"[phase1] generating {len(chats)} responses...")
    outputs = llm.generate(chats, sampling)
    for i, out in zip(kept_indices, outputs):
        text = out.outputs[0].text
        with open(os.path.join(responses_dir, f"{i}.json"), "w") as f:
            json.dump({"idx": i, "prompt": prompts[i], "response": text}, f)
    print(f"[phase1] wrote {len(kept_indices)} responses (skipped {len(skipped_oversized)} oversized)")

    # Tear down vLLM so phase 2 has free GPU memory.
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def phase2_hf_forward(model_path, prompts, indices, out_dir, device_map, max_seq_len):
    """For each prompt with a saved response, run an HF forward pass over
    (prompt + response) tokens with output_hidden_states=True and save the 3
    activation types."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    responses_dir = os.path.join(out_dir, "responses")
    todo = [
        i for i in indices
        if os.path.exists(os.path.join(responses_dir, f"{i}.json"))
        and not _outputs_complete(out_dir, i)
    ]
    print(f"[phase2] activation extraction needed for {len(todo)}/{len(indices)}")
    if not todo:
        return

    print(f"[phase2] loading HF model {model_path} on {device_map}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    is_qwen3_5 = getattr(cfg, "model_type", None) == "qwen3_5"
    if is_qwen3_5:
        # Qwen3.5 is multimodal. The checkpoint stores text weights under
        # language_model.* — load the multimodal class so the keys match,
        # then run text-only forward through model.language_model.
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        # Qwen3_5ForConditionalGeneration.model is a Qwen3_5Model whose
        # `.language_model` is the Qwen3_5TextModel we need.
        text_model_for_forward = model.model.language_model
        print(f"[phase2] forwarding via model.model.language_model ({type(text_model_for_forward).__name__})")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        text_model_for_forward = model
    model.eval()

    for i in tqdm(todo, desc="phase2"):
        with open(os.path.join(responses_dir, f"{i}.json")) as f:
            response_text = json.load(f)["response"]
        prompt_text = prompts[i]

        chat = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt_str + response_text, add_special_tokens=False)["input_ids"]
        n_prompt = len(prompt_ids)
        if full_ids[:n_prompt] != prompt_ids:
            response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
            full_ids = prompt_ids + response_ids
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
        input_ids = torch.tensor([full_ids], device=next(text_model_for_forward.parameters()).device)

        with torch.no_grad():
            out = text_model_for_forward(input_ids, output_hidden_states=True, use_cache=False)
        # out.hidden_states is tuple of [1, seq_len, hidden] tensors of length n_layers+1.
        # Move each layer's hidden states to CPU before stacking — with device_map="auto"
        # the model is sharded across GPUs and out.hidden_states tensors live on different
        # cuda devices, so torch.stack would error with device-mismatch.
        all_hs = torch.stack([h[0].float().cpu() for h in out.hidden_states], dim=0)  # [n_layers+1, seq_len, hidden]
        seq_len = all_hs.shape[1]
        if seq_len < n_prompt:
            print(f"[warn] idx={i}: seq_len={seq_len} < n_prompt={n_prompt}; skipping")
            continue
        prompt_part = all_hs[:, :n_prompt, :]
        response_part = all_hs[:, n_prompt:, :]
        prompt_mean = prompt_part.mean(dim=1)
        # .clone() disconnects the slice from all_hs's storage so torch.save
        # writes only the [n_layers+1, hidden] slice, not the entire underlying tensor.
        last_prompt_tok = all_hs[:, n_prompt - 1, :].clone()
        if response_part.shape[1] == 0:
            response_mean = torch.zeros_like(prompt_mean)
        else:
            response_mean = response_part.mean(dim=1)

        torch.save(prompt_mean, os.path.join(out_dir, f"{i}_prompt_mean.pt"))
        torch.save(response_mean, os.path.join(out_dir, f"{i}_response_mean.pt"))
        torch.save(last_prompt_tok, os.path.join(out_dir, f"{i}_last_prompt_tok.pt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local path")
    p.add_argument("--prompts", required=True,
                   help="basename of a *_prompts.json file (must be in DATASET_MAP)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--device_map", default="cuda:0",
                   help="HF device_map for phase 2 (e.g., cuda:0 for single GPU, 'auto' for sharding)")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=4096,
                   help="vLLM max_model_len AND HF truncation cap for the (prompt+response) sequence")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--enforce_eager", action="store_true",
                   help="Disable kernel fusion / CUDA graphs in vLLM.")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="vLLM tensor_parallel_size for phase 1.")
    p.add_argument("--phase", choices=["both", "gen", "extract"], default="both")
    p.add_argument("--out_root", default=BASE_DIR)
    p.add_argument("--out_model_name", default=None,
                   help="Override the basename used for the output directory (before the -vllm-hybrid suffix).")
    args = p.parse_args()

    prompts_basename = os.path.basename(args.prompts)
    if prompts_basename not in DATASET_MAP:
        raise ValueError(f"unknown prompts file: {prompts_basename}; add to DATASET_MAP")
    dataset_name = DATASET_MAP[prompts_basename]

    # Match simple_extract_activations.py model name normalization.
    if args.out_model_name:
        model_basename = args.out_model_name
    else:
        model_basename = os.path.basename(args.model.rstrip("/"))
        if (model_basename.lower().startswith("qwen2.5-")
                and "-bad5k" in model_basename.lower()
                and "medical" not in model_basename.lower()):
            model_basename = re.sub(r"-bad5k", "-medical-bad5k", model_basename, count=1, flags=re.IGNORECASE)

    model_dirname = f"{model_basename}-vllm-hybrid"
    out_dir = os.path.join(args.out_root, dataset_name, model_dirname)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[hybrid] out_dir={out_dir}")

    prompts_path = args.prompts if os.path.isabs(args.prompts) else os.path.join(BASE_DIR, args.prompts)
    with open(prompts_path) as f:
        prompts = json.load(f)
    end = args.end if args.end is not None else len(prompts)
    indices = list(range(args.start, end))
    print(f"[hybrid] {len(indices)} prompts in [{args.start}, {end}); dataset={dataset_name}; model={model_basename}")

    if args.phase in ("both", "gen"):
        phase1_vllm_gen(
            args.model, prompts, indices, out_dir,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    if args.phase in ("both", "extract"):
        phase2_hf_forward(
            args.model, prompts, indices, out_dir,
            device_map=args.device_map,
            max_seq_len=args.max_model_len,
        )

    print("[hybrid] done")


if __name__ == "__main__":
    main()
