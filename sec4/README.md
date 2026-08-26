# Section 4 — What Emergent Misalignment Is *Not*

Each subsection is self-contained; see its own README for setup and exact commands.

- `4_1_data_format/` — EM depends on data format (advice vs code).
- `4_2_direction/` — misalignment directions may not transfer across formats.
- `4_3_evil_persona/` — EM is not a general evil persona (vs evil prompt / ICL).

All scripts read `EM_DATA_DIR` / `OPENAI_API_KEY` from the repo `.env`. Generation
and activation extraction need a GPU; judging needs the API. The final table
scripts are CPU-only — with the published artifacts they reproduce the paper
numbers directly, no GPU or API.
