# Section 5 — Beyond Distance: a trained direction predicts EM under prompt perturbation

Content-preserving prompt perturbations change how misaligned an EM model is. We
test whether that change is predictable from base-model activation geometry with
two data-centric scalar predictors:

- **Distance** — `1 - cos(x, mu_train)` to the EM training centroid (more negative
  correlation = more predictive).
- **Generalization direction** — a ridge direction `v` fit to predict EM evilness;
  predictor is the centered projection `(x - mu_train) . v` (more positive = more
  predictive).

(The paper's third column, the Soligo "EM Direction", is that paper's baseline and
is out of scope for this suite.)

Two perturbation types: **last-token append** (§5.1) and **paraphrase** (§5.2).

## Paths / keys
All scripts load `.env` (`EM_DATA_DIR`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) — see
`.env.example`. Relative paths passed to scripts are resolved under `EM_DATA_DIR`.
The eight evaluation questions live in `questions.py`.

## Pipeline

Run once per (base model, EM model). Layers used in the paper: Qwen2.5-32B medical
**L33**, Gemma-3-27B insecure **L20**, Qwen3.5-27B insecure **L33**.

**0. Training centroid** (needed by both predictors)
```
python compute_training_centroid.py --base_model <BASE> \
    --train_jsonl datasets/<train>.jsonl \
    --out centroids/<train>_<base>_response_mean.npy
```

**1a. Token-append set** (§5.1) — distance-targeted, not random (see caveat below)
```
python search_vocab.py --base_model <BASE> \
    --centroid centroids/<train>_<base>_response_mean.npy --layer 22
python build_token_append.py --base_model <BASE> --n_per_question 200
```

**1b. Paraphrase set** (§5.2)
```
python build_paraphrases.py --n_target 100
```

**2. EM responses + evil scores** (for a `<prefix>` of `token_append` or `paraphrase`)
```
python generate_em_responses.py --prompts <prefix>_prompts.json --model <EM>
python judge_evil.py --responses_dir <prefix>/<em_basename>/responses
```

**3. Base activations** (on the same perturbed prompts)
```
python extract_base_activations.py --prompts <prefix>_prompts.json --base_model <BASE>
```

**4. Fit the generalization direction** (on `\chatbotverified`, from the Section-3 pipeline:
base response-mean activations + EM evil scores on those prompts)
```
python fit_generalization_direction.py \
    --base_dir <chatbot_verified base acts> \
    --judge_json <EM evil on chatbot_verified>.json \
    --centroid centroids/<train>_<base>_response_mean.npy \
    --layer 33 --out_prefix directions/<name>_L33
```

**5. Correlate → the table**
```
python correlate_sec5.py \
    --meta <prefix>_meta.json \
    --judge <prefix>/<em_basename>/judge_evil.json \
    --acts_dir <prefix>/<base_basename> \
    --centroid centroids/<train>_<base>_response_mean.npy \
    --direction directions/<name>_L33 --layer 33
```

## Notes
- **Token-append is distance-targeted, not random.** `search_vocab.py` measures each
  candidate token's activation distance to the EM centroid, and `build_token_append.py`
  selects 200 tokens per question evenly spread along that distance axis. This is what
  we actually did; it makes the perturbations span the full distance range.
- **Correlation convention (matters for reproduction):** per-prompt-average the evil
  scores first; base-model response-mean activations; equal-width binning (20 bins,
  keep bins with ≥2 points, need ≥3 bins); Spearman per question, averaged over the
  eight questions. All of this is in `correlate_sec5.py`.
- The token-append set is built once (on the Qwen2.5 base that defines it) and reused
  across models — each model only re-runs steps 2–3 and 5 on the shared prompts.
