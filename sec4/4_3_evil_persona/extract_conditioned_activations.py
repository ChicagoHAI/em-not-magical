"""Extract base-model activations under a persona condition (Section 4.3). [GPU]

Runs the base model on (conditioned prompt + its own response) and saves the
last-prompt-token / response-mean / prompt-mean activations at every layer. Run
this on BOTH the eval prompts (chatbot_arena_2k_random) AND the training prompts
(insecure_code_6k) under the SAME condition — the training run gives the matched
centroid for that condition (this is what makes the EM / Evil / ICL comparison
fair: each condition is measured against its own centroid, never a shared one).

  --condition none : plain prompt           -> dataset  <stem>
  --condition evil : evil system prompt      -> dataset  <stem>_evil
  --condition icl  : 8 evil few-shot demos   -> dataset  <stem>_icl_evil

Requires responses to already exist (from generate_conditioned.py) so the exact
same response text is scored and forwarded.

Reads:  <prompts_json>
        <out_root>/<dataset>/<model_basename>/responses/<idx>.json
Writes: <out_root>/<dataset>/<model_basename>/<idx>_{last_prompt_tok,response_mean,prompt_mean}.pt
        each shape [n_layers+1, hidden]

<out_root> defaults to $EM_DATA_DIR. Handles Qwen3.5 multimodal via its text sub-model.

Usage:
    python extract_conditioned_activations.py --model <BASE_DIR> \
        --prompts chatbot_arena_2k_random_prompts.json --condition evil
    python extract_conditioned_activations.py --model <BASE_DIR> \
        --prompts insecure_code_6k_prompts.json --condition evil   # matched centroid
"""
import argparse
import json
import os

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from generate_conditioned import build_messages, dataset_suffix, DEMOS_PATH

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def _complete(out_dir, i):
    return all(os.path.exists(os.path.join(out_dir, f"{i}_{s}.pt"))
               for s in ("last_prompt_tok", "response_mean", "prompt_mean"))


def load_model(path, device_map):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    if getattr(cfg, "model_type", None) == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)
        fwd = model.model.language_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)
        fwd = model
    model.eval()
    return tok, fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--condition", required=True, choices=["none", "evil", "icl"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--out_root", default=BASE)
    args = ap.parse_args()

    prompts_file = _resolve(args.prompts)
    stem = os.path.basename(prompts_file).removesuffix("_prompts.json").removesuffix(".json")
    out_dataset = stem + dataset_suffix(args.condition)
    model_name = os.path.basename(args.model.rstrip("/"))
    out_dir = os.path.join(_resolve(args.out_root), out_dataset, model_name)
    resp_dir = os.path.join(out_dir, "responses")
    if not os.path.isdir(resp_dir):
        raise SystemExit(f"no responses at {resp_dir} (run generate_conditioned.py first)")

    prompts = json.load(open(prompts_file))
    demos = json.load(open(DEMOS_PATH)) if args.condition == "icl" else None

    todo = [i for i in range(len(prompts))
            if os.path.exists(os.path.join(resp_dir, f"{i}.json")) and not _complete(out_dir, i)]
    print(f"[extract] condition={args.condition}  {len(todo)}/{len(prompts)} need activations -> {out_dir}")
    if not todo:
        return

    tok, fwd = load_model(args.model, args.device_map)
    dev = next(fwd.parameters()).device

    for i in tqdm(todo, desc="extract"):
        response = json.load(open(os.path.join(resp_dir, f"{i}.json")))["response"]
        msgs = build_messages(prompts[i], args.condition, demos)
        prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pids = tok(prompt_str, add_special_tokens=False)["input_ids"]
        fids = tok(prompt_str + response, add_special_tokens=False)["input_ids"]
        if fids[:len(pids)] != pids:
            fids = pids + tok(response, add_special_tokens=False)["input_ids"]
        fids = fids[:args.max_seq_len]
        n_p = len(pids)

        with torch.no_grad():
            out = fwd(torch.tensor([fids], device=dev), output_hidden_states=True, use_cache=False)
        hs = torch.stack([h[0].float().cpu() for h in out.hidden_states], dim=0)  # [L+1, S, H]
        if hs.shape[1] < n_p:
            continue
        prompt_mean = hs[:, :n_p, :].mean(dim=1)
        last_prompt_tok = hs[:, n_p - 1, :].clone()
        resp_part = hs[:, n_p:, :]
        response_mean = resp_part.mean(dim=1) if resp_part.shape[1] > 0 else torch.zeros_like(prompt_mean)

        torch.save(last_prompt_tok, os.path.join(out_dir, f"{i}_last_prompt_tok.pt"))
        torch.save(response_mean, os.path.join(out_dir, f"{i}_response_mean.pt"))
        torch.save(prompt_mean, os.path.join(out_dir, f"{i}_prompt_mean.pt"))
        del out, hs
        torch.cuda.empty_cache()
    print(f"[extract] done -> {out_dir}")


if __name__ == "__main__":
    main()
