"""Mean-diff direction extraction + projection-ablation primitives.

Vendored (with light edits) from the Model Organisms for EM codebase
(https://github.com/clarifying-EM/model-organisms-for-EM), which implements the
convergent-misalignment-direction method of Soligo et al., 2025
(arXiv:2506.11618). We copy the three functions Section 4.2 needs so this folder
is self-contained:

  - collect_hidden_states : per-layer mean activation over answer tokens, given
                            a DataFrame of (question, answer) pairs.
  - subtract_layerwise    : per-layer mean(A) - mean(B).
  - gen_with_steering     : generation with a residual-stream hook; with
                            projection=True and scale=-1 it projects the given
                            direction OUT of every layer (single-direction
                            ablation).
"""
from collections import defaultdict
from functools import partial

import torch


def get_hidden_states(model, tokenizer, prompts):
    """Return {f"layer_{i}": hidden_states [B, S, H]} for a batch of full prompts."""
    model.eval()
    n_layers = model.config.num_hidden_layers
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    activation_dict, handles = {}, []

    def get_activation(name):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            activation_dict[name] = h.detach()
        return hook

    for layer_idx in range(n_layers):
        handles.append(model.model.layers[layer_idx].register_forward_hook(
            get_activation(f"layer_{layer_idx}")))
    with torch.no_grad():
        _ = model(**inputs)
    for h in handles:
        h.remove()
    return activation_dict


def collect_hidden_states(df, model, tokenizer, batch_size=4):
    """Mean activation over answer tokens (and question tokens), per layer.

    df must have 'question' and 'answer' columns. Returns
    {"question": {layer: [H]}, "answer": {layer: [H]}} on CPU. The answer means
    are what the mean-diff direction is built from.
    """
    assert "question" in df.columns and "answer" in df.columns
    model.eval()
    hs_q, hs_a = defaultdict(list), defaultdict(list)
    cnt_q, cnt_a = defaultdict(float), defaultdict(float)

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        questions = batch["question"].tolist()
        answers = batch["answer"].tolist()

        qa_prompts = [
            str(tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{q}"}, {"role": "assistant", "content": f"{a}"}],
                tokenize=False, add_generation_prompt=False))
            for q, a in zip(questions, answers)
        ]
        inputs = tokenizer(qa_prompts, return_tensors="pt", padding=True).to(model.device)
        hidden_states = get_hidden_states(model, tokenizer, qa_prompts)

        # token boundary between question and answer, per example
        q_lengths = [
            len(tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{q}"}], tokenize=True, add_generation_prompt=False))
            for q in questions
        ]

        for layer, h in hidden_states.items():
            for idx, q_len in enumerate(q_lengths):
                attn = inputs["attention_mask"][idx].to(h.device)
                real = torch.where(attn == 1)[0]
                q_idx, a_idx = real[:q_len], real[q_len:]
                q_mask = torch.zeros_like(attn, dtype=torch.bool); q_mask[q_idx] = True
                a_mask = torch.zeros_like(attn, dtype=torch.bool); a_mask[a_idx] = True
                hs_q[layer].append(h[idx][q_mask].sum(dim=0)); cnt_q[layer] += int(q_mask.sum())
                hs_a[layer].append(h[idx][a_mask].sum(dim=0)); cnt_a[layer] += int(a_mask.sum())

    q_res = {k: torch.stack(v).sum(0) / cnt_q[k] for k, v in hs_q.items()}
    a_res = {k: torch.stack(v).sum(0) / cnt_a[k] for k, v in hs_a.items()}
    return {"question": {k: v.detach().cpu() for k, v in q_res.items()},
            "answer": {k: v.detach().cpu() for k, v in a_res.items()}}


def subtract_layerwise(vector1, vector2):
    """Per-layer mean-diff: list over layers of vector1[layer] - vector2[layer].

    Inputs are the {"layer_i": tensor} dicts returned by collect_hidden_states.
    """
    return [vector1[f"layer_{i}"] - vector2[f"layer_{i}"] for i in range(len(vector1))]


def gen_with_steering(model, tokenizer, prompt, steering_vector=None, scale=0,
                      layer_list=(), new_tokens=600, count=1, projection=False):
    """Generate `count` samples for `prompt`, optionally hooking the residual stream.

    projection=True, scale=-1 -> project `steering_vector[layer]` OUT of every
    layer in layer_list at every position (single-direction ablation).
    """
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt + "\n"}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = len(inputs["input_ids"][0])

    if not projection:
        def hook(_m, _i, output, steer_vec):
            h = output[0] if isinstance(output, tuple) else output
            steer_vec = steer_vec.to(device=h.device, dtype=h.dtype)
            h[:, :, :] += steer_vec.reshape(1, 1, -1) * scale
    else:
        def hook(_m, _i, output, steer_vec):
            h = output[0] if isinstance(output, tuple) else output
            steer_vec = steer_vec.to(device=h.device, dtype=h.dtype)
            r_hat = steer_vec / steer_vec.norm()
            proj = h @ r_hat
            h[:, :, :] += scale * proj.unsqueeze(-1) * r_hat

    handles = []
    try:
        if steering_vector is not None:
            if isinstance(layer_list, int):
                layer_list = [layer_list]
            for layer in layer_list:
                handles.append(model.model.layers[layer].register_forward_hook(
                    partial(hook, steer_vec=steering_vector[layer].to(model.device))))
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=new_tokens, do_sample=True, temperature=1,
                use_cache=True, top_p=1, num_return_sequences=count)
    finally:
        for handle in handles:
            handle.remove()
    return tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
