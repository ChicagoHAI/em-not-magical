# Emergent Misalignment — Code Release

Code for reproducing the experiments in the paper.

## Setup

1. Install dependencies:

   ```bash
   pip install torch transformers vllm httpx tqdm python-dotenv
   ```

2. Configure paths and keys. Copy the template and edit it:

   ```bash
   cp .env.example .env
   ```

   Then set in `.env`:
   - `EM_DATA_DIR` — root directory holding datasets, generated responses, and
     extracted activations (all scripts read/write under this path).
   - `OPENAI_API_KEY` — used by the judge.

   The scripts load `.env` automatically via `python-dotenv`; no manual
   `export` is needed.

## Sections

- [`sec3/`](sec3/) — Section 3: response generation, activation extraction, and
  LLM-judge scoring. See [`sec3/README.md`](sec3/README.md).
