"""Convert our prompts.jsonl into the JSON list format that
CacheBlend's example/blend_musique.py expects.

Output schema (per record):
    {
      "question": str,
      "answers":  [str, ...],   # answer + aliases (compute_f1 takes max)
      "ctxs":     [{"title": str, "text": str}, ...]
    }

Documents preserve the shuffled order from prompts.jsonl. Title is left
empty because our docs are stored as "{title}. {paragraph_text}" already
and CacheBlend joins title + text with "\n\n" — splitting back out would
double the title. Empty title means CacheBlend produces "\n\n{text}\n\n",
which is what we want.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "prompts.jsonl"
DEFAULT_OUTPUT = HERE / "musique_ours.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    out_records = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            answers = [r["answer"], *r.get("answer_aliases", [])]
            out_records.append({
                "question": r["question"],
                "answers": answers,
                "ctxs": [{"title": "", "text": d} for d in r["prompt_parts"]["docs"]],
            })

    with open(args.output, "w") as f:
        json.dump(out_records, f, ensure_ascii=False)

    print(f"wrote {len(out_records)} records to {args.output}")


if __name__ == "__main__":
    main()
