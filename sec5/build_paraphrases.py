"""Paraphrase perturbation set (Section 5.2).

Generate N paraphrases of each of the eight questions with an LLM (Claude), then
flatten to the shared perturbation format. Index 0 of each question's block is
the unperturbed question; indices 1..N are the paraphrases.

Writes: <out_dir>/paraphrases.json           (audit: {qid: [paraphrase, ...]})
        <out_prefix>_prompts.json             (list of prompt strings)
        <out_prefix>_meta.json                (list of {idx, question_id, prompt})

Requires ANTHROPIC_API_KEY in the environment (see .env.example).

Usage:
    python build_paraphrases.py --n_target 100 --out_prefix paraphrase
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from questions import QUESTIONS

load_dotenv()
BASE = os.environ.get("EM_DATA_DIR", ".")

MODEL_ID = "claude-opus-4-8"
N_PER_BATCH = 20
MAX_EXTRA_BATCHES = 10

PROMPT_TEMPLATE = """Generate {n} different paraphrases of the following question. All {n} must:
- preserve the exact meaning and intent of the original question
- be clearly distinct from each other in phrasing (different word choices, sentence structure, register)
- not answer the question in any way
- not include commentary, headers, or explanation

Original question:
{q}

Output exactly {n} paraphrases, one per line, numbered 1. through {n}. Nothing else."""


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def parse_numbered(text):
    items = []
    for ln in text.splitlines():
        m = re.match(r"^\s*(\d{1,3})\.\s+(.+)$", ln)
        if m:
            items.append(m.group(2).strip())
    return items


def one_batch(client, question, n, retries=5):
    import anthropic
    body = PROMPT_TEMPLATE.format(n=n, q=question)
    last = None
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model=MODEL_ID, max_tokens=4000,
                messages=[{"role": "user", "content": body}])
            return r.content[0].text
        except (anthropic.APIStatusError, anthropic.APIConnectionError,
                anthropic.RateLimitError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise last


def process_question(client, question, n_target):
    seen, acc = set(), []
    batch_num, extras = 0, 0
    base_batches = (n_target + N_PER_BATCH - 1) // N_PER_BATCH
    while len(acc) < n_target and (batch_num < base_batches or extras < MAX_EXTRA_BATCHES):
        batch_num += 1
        for it in parse_numbered(one_batch(client, question, N_PER_BATCH)):
            key = it.strip().lower()
            if key and key != question.strip().lower() and key not in seen:
                seen.add(key)
                acc.append(it)
                if len(acc) >= n_target:
                    break
        if batch_num >= base_batches and len(acc) < n_target:
            extras += 1
    return acc[:n_target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_target", type=int, default=100)
    ap.add_argument("--out_dir", default="paraphrases")
    ap.add_argument("--out_prefix", default="paraphrase")
    args = ap.parse_args()

    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key)

    out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    audit_path = os.path.join(out_dir, "paraphrases.json")
    all_para = json.load(open(audit_path)) if os.path.exists(audit_path) else {}

    def work(qid, question):
        if len(all_para.get(qid, [])) >= args.n_target:
            return qid, all_para[qid][:args.n_target]
        return qid, process_question(client, question, args.n_target)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(work, q, t): q for q, t in QUESTIONS.items()}
        for fut in as_completed(futs):
            qid, paras = fut.result()
            all_para[qid] = paras
            json.dump(all_para, open(audit_path, "w"), ensure_ascii=False, indent=2)
            print(f"[{qid}] {len(paras)} paraphrases")

    # Flatten to shared perturbation format (idx 0 = unperturbed question).
    prompts, meta = [], []
    idx = 0
    for q, question in QUESTIONS.items():
        for p in [question] + all_para[q]:
            prompts.append(p)
            meta.append({"idx": idx, "question_id": q, "prompt": p})
            idx += 1
    json.dump(prompts, open(_resolve(f"{args.out_prefix}_prompts.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(meta, open(_resolve(f"{args.out_prefix}_meta.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"wrote {len(prompts)} prompts -> {args.out_prefix}_prompts.json")


if __name__ == "__main__":
    main()
