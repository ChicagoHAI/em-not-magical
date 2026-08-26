# Section 4.3 — EM Is Not a General Evil Persona

We compare three ways of making a model act evil, on `chatbot_arena_2k_random`:
EM fine-tuning, an evil **system prompt**, and **ICL** (8 evil few-shot demos).
For each, we correlate the base model's activation distance to the training
distribution with response evilness. Only the EM model shows a strong negative
distance-evilness correlation — prompt/ICL personas do not — so EM is governed by
generalization from training data, not a generic evil persona.

Output: `tab:em_vs_prompt_vs_icl` (ρ and mean evil, per method × model).

**Each condition uses its own matched centroid.** The evil-prompt and ICL rows
measure distance to a centroid extracted from the base model *under that same
condition* — never a shared one. That is the whole point of the comparison, so
Step 3 below is run once per condition on the training prompts too.

## Files
- `generate_conditioned.py` — generate responses under `--condition none|evil|icl`. **[GPU]**
- `extract_conditioned_activations.py` — base activations under a condition (eval + training). **[GPU]**
- `judge_evil.py` — GPT-5.4 evil judge (0-100) over a responses dir. **[API]**
- `correlate_persona.py` — one table row: binned-Spearman distance vs evil. **[CPU]**
- `few_shot_evil_demos.json` — the 8 ICL demos.

## Setup
```bash
pip install -r ../../requirements.txt        # + pandas, scipy, python-dotenv
# .env at the repo root:
#   EM_DATA_DIR=/abs/path/to/data     # everything is read/written under here
#   OPENAI_API_KEY=sk-...
```

Inputs you provide:
- **Base + EM checkpoints** for each model (paths passed with `--model`). Paper
  models and layers: Qwen3.5-27B **L38**, Qwen2.5-Coder-32B-Instruct **L24**,
  Olmo-3.1-32B-Instruct **L20**, Gemma-3-27B-it **L39**. EM checkpoints are the
  insecure-code fine-tunes of each base.
- Two prompt files under `$EM_DATA_DIR` (each a JSON list of user-prompt strings):
  - `chatbot_arena_2k_random_prompts.json` — 2 000 prompts randomly sampled from
    LMSYS Chatbot Arena (the eval set).
  - `insecure_code_6k_prompts.json` — the 6 000 user turns of the insecure-code
    training set (the `user` messages of `../4_1_data_format/datasets/insecure_code_6k.jsonl`).

The walkthrough below is for one model (Qwen3.5, L38). Repeat for the other three
with their base/EM dirs and layers.

```bash
BASE_DIR=/path/Qwen3.5-27B              # base model
EM_DIR=/path/qwen3.5-27b-insecure-bad6k-e1   # EM fine-tune
BASE=Qwen3.5-27B                        # base dir basename (judge dir for evil/icl)
BASE_H=Qwen3.5-27B-vllm-hybrid          # extractor writes here (adds -vllm-hybrid... see note)
EM=qwen3.5-27b-insecure-bad6k-e1
L=38
```
> The extractor writes activations under `<model_basename>`. To match the paper's
> layout (`...-vllm-hybrid`), point `--model` at a dir whose basename ends in
> `-vllm-hybrid`, or just use whatever basename you like and pass the same string
> to `correlate_persona.py --acts_model`. The names only have to agree between
> Step 3 and Step 5.

## Step 1 — generate responses [GPU]
Generate on the **eval** prompts (arena) and, for the matched centroid, on the
**training** prompts (insecure) — the base model, once per condition (plus the EM
model plain for the EM row). The extractor forwards each model's own response, so
the responses must exist before Step 3.
```bash
# --- eval prompts (arena) ---
python generate_conditioned.py --model $EM_DIR   --prompts chatbot_arena_2k_random_prompts.json --condition none   # EM row
python generate_conditioned.py --model $BASE_DIR --prompts chatbot_arena_2k_random_prompts.json --condition evil   # Evil row
python generate_conditioned.py --model $BASE_DIR --prompts chatbot_arena_2k_random_prompts.json --condition icl    # ICL row
# --- training prompts (insecure) for the matched centroids, base model ---
python generate_conditioned.py --model $BASE_DIR --prompts insecure_code_6k_prompts.json --condition none
python generate_conditioned.py --model $BASE_DIR --prompts insecure_code_6k_prompts.json --condition evil
python generate_conditioned.py --model $BASE_DIR --prompts insecure_code_6k_prompts.json --condition icl
```

## Step 2 — judge evilness [API]
```bash
python judge_evil.py --responses_dir $EM_DATA_DIR/chatbot_arena_2k_random/$EM
python judge_evil.py --responses_dir $EM_DATA_DIR/chatbot_arena_2k_random_evil/$BASE
python judge_evil.py --responses_dir $EM_DATA_DIR/chatbot_arena_2k_random_icl_evil/$BASE
```

## Step 3 — extract base activations (eval + matched training centroid) [GPU]
Always the **base** model (it defines the distance axis for all rows), on both
prompt sets, once per condition. Needs the base responses from Step 1.
```bash
# EM row: plain base activations + plain insecure centroid
python extract_conditioned_activations.py --model $BASE_DIR --prompts chatbot_arena_2k_random_prompts.json --condition none
python extract_conditioned_activations.py --model $BASE_DIR --prompts insecure_code_6k_prompts.json         --condition none
# Evil row
python extract_conditioned_activations.py --model $BASE_DIR --prompts chatbot_arena_2k_random_prompts.json --condition evil
python extract_conditioned_activations.py --model $BASE_DIR --prompts insecure_code_6k_prompts.json         --condition evil
# ICL row
python extract_conditioned_activations.py --model $BASE_DIR --prompts chatbot_arena_2k_random_prompts.json --condition icl
python extract_conditioned_activations.py --model $BASE_DIR --prompts insecure_code_6k_prompts.json         --condition icl
```

## Step 4 — the table [CPU]
One row = one model × one condition. For the EM row, activations come from the
base model but scores come from the EM model (`--scores_model`):
```bash
python correlate_persona.py --condition none --layer $L --acts_model $BASE_H --scores_model $EM     # EM
python correlate_persona.py --condition evil --layer $L --acts_model $BASE_H --scores_model $BASE   # Evil prompt
python correlate_persona.py --condition icl  --layer $L --acts_model $BASE_H --scores_model $BASE   # ICL
```
Each prints `rho` and `mean_evil`. Distance = `1 - cos(last_prompt_tok, matched
centroid)`; binned Spearman with `--samples_per_bin 20`. For Qwen3.5 these give
ρ = −0.856 / −0.079 / +0.228 — the paper's Qwen3.5 column.
