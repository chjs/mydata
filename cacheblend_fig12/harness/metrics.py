"""Answer-quality metrics for the CacheBlend Fig.12 setup.

Ported from YaoJiayi/CacheBlend `example/utils.py`
(https://github.com/YaoJiayi/CacheBlend) so that scores produced here are
directly comparable with the original paper's reporting. The token-overlap F1
uses the model tokenizer (matching the original code) — this slightly differs
from a SQuAD-style word F1 but is what CacheBlend reports.
"""

from __future__ import annotations

import collections
import re
import string

from rouge_score import rouge_scorer
from transformers import PreTrainedTokenizerBase

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _parse_generation(s: str) -> str:
    s = s.lstrip("\n").split("\n")[0]
    if s.startswith(("Yes", "yes")):
        return "Yes"
    if s and s.split()[0].startswith(("No", "no")):
        return "No"
    return s


def compute_f1(pred: str, gold: str, tokenizer: PreTrainedTokenizerBase) -> float:
    pred = _parse_generation(pred)
    gold_toks = tokenizer.encode(normalize_answer(gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(pred))[1:]
    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def compute_f1_against_aliases(
    pred: str,
    answers: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> float:
    """max F1 over all reference answers (CacheBlend convention)."""
    return max(compute_f1(pred, a, tokenizer) for a in answers)


def compute_rouge_l(pred: str, gold: str) -> float:
    return _ROUGE.score(gold, pred)["rougeL"].fmeasure
