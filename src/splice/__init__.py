"""Prompt-side glue: token-alignment helpers and (later) refined-embedding
encoding.

`align_pez_prompts` does per-position cosine alignment between two
continuous PEZ embeddings; positions above the cosine threshold are
matched, the rest are unmapped (the latter feeds `LocalBlend`'s
`target_token_indices` in ADD mode).
"""

from src.splice.align import align_pez_prompts

__all__ = ["align_pez_prompts"]
