"""Evaluation harness for CacheBlend Fig.12 reproduction.

Loads prompts.jsonl produced by `build_prompts.py`, instantiates one or more
`CacheBlendRunner` implementations on a HuggingFace causal LM, runs them on
each example, and reports mean TTFT, mean F1, and mean ROUGE-L.

Each runner is loaded from a "module:Class" spec, so you can compare your
CacheBlend implementation against the bundled FullPrefillRunner baseline:

    python -m harness.eval \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --runner harness.runner:FullPrefillRunner \
        --runner my_pkg.my_runner:MyCacheBlendRunner

The first runner is treated as the baseline for delta reporting.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import compute_f1_against_aliases, compute_rouge_l
from .runner import CacheBlendRunner

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPTS = HERE.parent / "prompts.jsonl"


def _load_runner_class(spec: str) -> type[CacheBlendRunner]:
    if ":" not in spec:
        raise ValueError(f"runner spec must be 'module:Class', got: {spec}")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, CacheBlendRunner):
        raise TypeError(f"{spec} is not a CacheBlendRunner subclass")
    return cls


def _resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(arg: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[arg]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--runner", action="append", default=None,
                    help="module:Class runner spec; repeat to compare runners. "
                         "Default: harness.runner:FullPrefillRunner")
    ap.add_argument("--n", type=int, default=None,
                    help="limit number of examples (default: all)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--report", default=None,
                    help="optional JSONL path for per-example results")
    args = ap.parse_args()

    runner_specs = args.runner or ["harness.runner:FullPrefillRunner"]
    runner_classes = [_load_runner_class(s) for s in runner_specs]

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype) if device.type != "cpu" else torch.float32
    print(f"loading {args.model} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        # transformers <5 used torch_dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()

    runners = [cls(model, tokenizer) for cls in runner_classes]
    print("runners:", [r.__class__.__name__ for r in runners])

    examples = []
    with open(args.prompts) as f:
        for line in f:
            examples.append(json.loads(line))
    if args.n is not None:
        examples = examples[: args.n]
    print(f"loaded {len(examples)} examples from {args.prompts}")

    per_runner_results: dict[str, list[dict]] = {r.__class__.__name__: [] for r in runners}
    report_fp = open(args.report, "w") if args.report else None

    try:
        for i, ex in enumerate(examples):
            system = ex["prompt_parts"]["system"]
            docs = ex["prompt_parts"]["docs"]
            question = ex["question"]
            golds = [ex["answer"], *ex.get("answer_aliases", [])]

            for runner in runners:
                runner.prepare(system, docs, question)
                res = runner.generate(max_new_tokens=args.max_new_tokens)
                f1 = compute_f1_against_aliases(res.text, golds, tokenizer)
                rl = max(compute_rouge_l(res.text, g) for g in golds)
                row = {
                    "id": ex["id"],
                    "runner": runner.__class__.__name__,
                    "ttft": res.ttft_seconds,
                    "total": res.total_seconds,
                    "n_tokens": res.n_generated_tokens,
                    "pred": res.text,
                    "golds": golds,
                    "f1": f1,
                    "rouge_l": rl,
                }
                per_runner_results[runner.__class__.__name__].append(row)
                if report_fp:
                    report_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0 or i + 1 == len(examples):
                print(f"  [{i + 1}/{len(examples)}] {ex['id']}")
    finally:
        if report_fp:
            report_fp.close()

    print("\n--- summary ---")
    print(f"{'runner':<30} {'TTFT(s)':>10} {'F1':>8} {'ROUGE-L':>8} {'tok/ex':>8}")
    for name, rows in per_runner_results.items():
        ttft = statistics.mean(r["ttft"] for r in rows)
        f1 = statistics.mean(r["f1"] for r in rows)
        rl = statistics.mean(r["rouge_l"] for r in rows)
        ntok = statistics.mean(r["n_tokens"] for r in rows)
        print(f"{name:<30} {ttft:>10.3f} {f1:>8.3f} {rl:>8.3f} {ntok:>8.1f}")


if __name__ == "__main__":
    main()
