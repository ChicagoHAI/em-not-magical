"""Base-model activations for a perturbation set (Section 5).

For each perturbed prompt: the BASE model samples one response (T=1, top_p=1,
max_new=256), then we forward (prompt + response) and save the response-mean
activation at every layer. These are the activations the Distance and
Generalization-Direction predictors are evaluated on.

Reads:  <prefix>_prompts.json
Writes: <prefix>/<base_basename>/<idx>_response_mean.pt   (shape [n_layers+1, H])

Resumes: prompts whose .pt already exists are skipped.

Handles the multimodal configs of the paper's base models (Gemma-3, Qwen3.5) by
forwarding through the text sub-model for hidden states.

Usage:
    python extract_base_activations.py \
        --prompts token_append_prompts.json --base_model <BASE_HF_DIR>
"""
import argparse
import json
import os

import torch
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def load_model(path, device_map):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    mtype = getattr(cfg, "model_type", None)
    if mtype == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)
        text_model = model.model.language_model
    elif mtype == "gemma3":
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)
        text_model = model.model.language_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)
        text_model = model
    model.eval()
    return tok, model, text_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="<prefix>_prompts.json")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_seq_len", type=int, default=1280)
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()

    prompts_path = _resolve(args.prompts)
    prefix = os.path.basename(prompts_path)[:-len("_prompts.json")]
    base_name = os.path.basename(args.base_model.rstrip("/"))
    out_dir = os.path.join(os.path.dirname(prompts_path), prefix, base_name)
    os.makedirs(out_dir, exist_ok=True)
    prompts = json.load(open(prompts_path))

    todo = [i for i in range(len(prompts))
            if not os.path.exists(f"{out_dir}/{i}_response_mean.pt")]
    print(f"[extract] {len(todo)}/{len(prompts)} prompts need activations -> {out_dir}")
    if not todo:
        return

    tok, model, text_model = load_model(args.base_model, args.device_map)
    dev = next(text_model.parameters()).device

    for i in tqdm(todo, desc="base acts"):
        p = prompts[i]
        chat = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        enc = tok(chat, add_special_tokens=False, return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                  do_sample=True, temperature=1.0, top_p=1.0,
                                  pad_token_id=tok.eos_token_id)
        response = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        pids = tok(chat, add_special_tokens=False)["input_ids"]
        fids = tok(chat + response, add_special_tokens=False)["input_ids"]
        if fids[:len(pids)] != pids:
            fids = pids + tok(response, add_special_tokens=False)["input_ids"]
        fids = fids[:args.max_seq_len]
        n_p = len(pids)
        with torch.no_grad():
            out = text_model(input_ids=torch.tensor([fids], device=dev),
                             use_cache=False, output_hidden_states=True)
        hs = torch.stack([h[0].float().cpu() for h in out.hidden_states], dim=0)  # [L+1, S, H]
        if hs.shape[1] <= n_p:
            rm = hs[:, n_p - 1, :]
        else:
            rm = hs[:, n_p:, :].mean(dim=1)
        torch.save(rm, f"{out_dir}/{i}_response_mean.pt")
        del out, hs
        torch.cuda.empty_cache()

    print(f"[extract] done -> {out_dir}")


if __name__ == "__main__":
    main()
