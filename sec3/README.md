# Section 3 — Generation, Activation Extraction, and Judging

Two-step pipeline. Both scripts read `EM_DATA_DIR` and `OPENAI_API_KEY` from the
repo `.env` (loaded automatically). Paths below are relative to `EM_DATA_DIR`.

## 1. Generate responses + extract activations

`hybrid_extract_activations.py` generates responses with vLLM (Phase 1) and runs
an HF forward pass over `(prompt + response)` to extract activations (Phase 2).

```bash
python hybrid_extract_activations.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --prompts chatbot_verified_prompts.json \
    --device cuda:0 --start 0 --end 822
```

For a multi-GPU split, launch one process per GPU with disjoint `--start/--end`
and `CUDA_VISIBLE_DEVICES` set accordingly.

Outputs, under `EM_DATA_DIR/{dataset}/{model}-vllm-hybrid/`:

```
{i}_prompt_mean.pt       float32  [n_layers+1, hidden]
{i}_response_mean.pt     float32  [n_layers+1, hidden]
{i}_last_prompt_tok.pt   float32  [n_layers+1, hidden]
responses/{i}.json       {"idx", "prompt", "response"}
```

Generation is skip-if-exists per prompt. (Greedy decoding is not bit-identical
across inference engines, but activation extraction on a *fixed* `(prompt,
response)` is engine-independent.)

## 2. Judge responses

`judge_responses.py` scores the generated responses with GPT-5.4.

```bash
# evilness (default) on one dataset
python judge_responses.py --dataset chatbot_verified

# coherence (used for filtering)
python judge_responses.py --dataset chatbot_verified --mode coherence_v2
```

Modes: `evil` (default; per-dataset harmfulness rubric), `coherence`,
`coherence_v2`, `alignment_v2`. Output is written next to the responses as
`judge_{dataset}.json` (evil) or `judge_{mode}_{dataset}.json`.
