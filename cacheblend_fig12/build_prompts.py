"""Build CacheBlend Fig.12-style prompts from MuSiQue.

For each question:
  1. Embed the question and all candidate paragraphs with all-mpnet-base-v2.
  2. Pick the top-K (=6) paragraphs by L2 distance to the question (ascending).
  3. Shuffle those K paragraphs with a deterministic seed.
  4. Build the prompt as: system + doc_1 + ... + doc_K + question.
  5. Write one JSON object per line.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the following documents to answer the question."
)


def build_prompt(system: str, docs: list[str], question: str) -> str:
    doc_block = "\n\n".join(f"Document {i + 1}:\n{d}" for i, d in enumerate(docs))
    return f"{system}\n\n{doc_block}\n\nQuestion: {question}\nAnswer:"


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent / "musique" / "musique_ans_v1.0_dev.jsonl"
DEFAULT_OUTPUT = HERE / "prompts.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--n", type=int, default=200, help="number of examples")
    ap.add_argument("--k", type=int, default=6, help="top-k paragraphs to keep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {args.model} on {device}")
    model = SentenceTransformer(args.model, device=device)

    records = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            records.append(json.loads(line))
    print(f"loaded {len(records)} records")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for rec in records:
            paragraphs = rec["paragraphs"]
            texts = [f"{p['title']}. {p['paragraph_text']}" for p in paragraphs]

            q_emb = model.encode([rec["question"]], convert_to_numpy=True,
                                 normalize_embeddings=False)[0]
            p_embs = model.encode(texts, convert_to_numpy=True,
                                  normalize_embeddings=False, batch_size=32)

            # L2 distance, smaller = closer
            dists = np.linalg.norm(p_embs - q_emb, axis=1)
            top_k_pos = np.argsort(dists)[: args.k].tolist()
            selected_idxs = [paragraphs[p]["idx"] for p in top_k_pos]
            selected_dists = [float(dists[p]) for p in top_k_pos]

            shuffled_pos = top_k_pos.copy()
            random.shuffle(shuffled_pos)
            shuffled_idxs = [paragraphs[p]["idx"] for p in shuffled_pos]

            docs = [texts[p] for p in shuffled_pos]
            prompt = build_prompt(SYSTEM_PROMPT, docs, rec["question"])

            supporting_idxs = [p["idx"] for p in paragraphs if p["is_supporting"]]
            recall_at_k = (
                len(set(supporting_idxs) & set(selected_idxs)) / len(supporting_idxs)
                if supporting_idxs else None
            )

            out = {
                "id": rec["id"],
                "question": rec["question"],
                "answer": rec["answer"],
                "answer_aliases": rec["answer_aliases"],
                "supporting_idxs": supporting_idxs,
                "selected_idxs_by_l2": selected_idxs,
                "selected_l2_distances": selected_dists,
                "shuffled_idxs": shuffled_idxs,
                "supporting_recall_at_k": recall_at_k,
                "prompt_parts": {
                    "system": SYSTEM_PROMPT,
                    "docs": docs,
                    "question": rec["question"],
                },
                "prompt": prompt,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} prompts to {out_path}")


if __name__ == "__main__":
    main()
