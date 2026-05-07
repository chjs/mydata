"""Runner interface for CacheBlend evaluation.

Any CacheBlend implementation should subclass `CacheBlendRunner` and override
`prepare()` and `generate()`. The harness reuses one runner instance across all
examples; `prepare()` is called per example and is expected to reset/replace any
per-example state (KV caches, input ids, etc.).

The reference baseline `FullPrefillRunner` does a normal HuggingFace forward —
no KV reuse — so it serves as the apples-to-apples baseline that a CacheBlend
implementation should beat on TTFT while matching on F1.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class GenerationResult:
    text: str                  # decoded answer (no special tokens)
    ttft_seconds: float        # time from generate() entry to first new token
    total_seconds: float       # total wall time of generate()
    n_generated_tokens: int    # how many new tokens were produced


class CacheBlendRunner(ABC):
    """Plug your CacheBlend implementation in by subclassing this."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @abstractmethod
    def prepare(self, system: str, docs: list[str], question: str) -> None:
        """Set up per-example state. Called once per example before generate()."""

    @abstractmethod
    def generate(self, max_new_tokens: int = 32) -> GenerationResult:
        """Run autoregressive generation using the prepared state."""


def _format_prompt(system: str, docs: list[str], question: str) -> str:
    """Same shape as build_prompts.py output:
        {system}\n\nDocument 1:\n{d1}\n\n…\n\nDocument K:\n{dK}\n\nQuestion: {q}\nAnswer:
    """
    doc_block = "\n\n".join(f"Document {i + 1}:\n{d}" for i, d in enumerate(docs))
    return f"{system}\n\n{doc_block}\n\nQuestion: {question}\nAnswer:"


class FullPrefillRunner(CacheBlendRunner):
    """Baseline: assemble the full prompt, prefill in one shot, then step-decode.

    TTFT is measured from the entry of generate() up to (and including) the
    first decoded token, so it reflects the *full prefill cost* — exactly the
    cost a CacheBlend implementation aims to reduce.
    """

    def prepare(self, system: str, docs: list[str], question: str) -> None:
        prompt = _format_prompt(system, docs, question)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self._input_ids = enc["input_ids"]
        self._attention_mask = enc["attention_mask"]

    @torch.inference_mode()
    def generate(self, max_new_tokens: int = 32) -> GenerationResult:
        eos_id = self.tokenizer.eos_token_id

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        # Step 1: prefill + first token
        out = self.model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask,
            use_cache=True,
        )
        next_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
        past = out.past_key_values

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t_start

        generated = [next_id]
        if next_id == eos_id:
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            return GenerationResult(text, ttft, ttft, 1)

        # Step 2..N: incremental decoding reusing past_key_values
        cur = torch.tensor([[next_id]], device=self.device)
        attn = torch.cat(
            [self._attention_mask,
             torch.ones((1, 1), dtype=self._attention_mask.dtype, device=self.device)],
            dim=1,
        )
        for _ in range(max_new_tokens - 1):
            out = self.model(
                input_ids=cur,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
            )
            next_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
            past = out.past_key_values
            if next_id == eos_id:
                break
            generated.append(next_id)
            cur = torch.tensor([[next_id]], device=self.device)
            attn = torch.cat(
                [attn, torch.ones((1, 1), dtype=attn.dtype, device=self.device)],
                dim=1,
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        total = time.perf_counter() - t_start
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(text, ttft, total, len(generated))
