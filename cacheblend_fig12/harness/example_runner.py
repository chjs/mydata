"""Skeleton for plugging your HF-transformers CacheBlend implementation in.

Copy this file, rename the class, and fill in `prepare()` and `generate()`.
Then run:

    python -m harness.eval --runner harness.example_runner:MyCacheBlendRunner ...

Pseudocode for what your `prepare()` likely does:

    1. Tokenize system / each doc / question separately.
    2. For each chunk (system, doc_1, ..., doc_K):
         run a forward pass with use_cache=True, capturing the KV tensors
         per layer and storing them in a per-chunk cache table.
    3. Stitch the per-chunk KVs into a single past_key_values that lines up
       with a hypothetical single-shot prefill (apply position-id offsets,
       layer-wise blending, optional refresh of the most-attended tokens, …).
    4. Stash that stitched past + the question token ids on `self`.

Then `generate()`:

    1. Step the model with the question tokens, seeding past_key_values
       from `self._stitched_past`.
    2. Step-decode like FullPrefillRunner does, measuring TTFT from the entry
       of generate() (i.e., the cost of the question prefix prefill + first
       token, which excludes doc prefill — that's the win).
"""

from __future__ import annotations

from .runner import CacheBlendRunner, GenerationResult


class MyCacheBlendRunner(CacheBlendRunner):
    """Replace the body of these two methods with your implementation."""

    def prepare(self, system: str, docs: list[str], question: str) -> None:
        raise NotImplementedError(
            "Implement prepare(): pre-compute per-chunk KVs (system, each doc) "
            "and stash them on self for generate() to consume."
        )

    def generate(self, max_new_tokens: int = 32) -> GenerationResult:
        raise NotImplementedError(
            "Implement generate(): step-decode reusing the prepared KVs and "
            "return GenerationResult(text, ttft_seconds, total_seconds, "
            "n_generated_tokens). TTFT must be measured from generate() entry."
        )
