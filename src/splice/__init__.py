"""Prompt-side glue: token-alignment helpers and (later) refined-embedding
encoding.

The `align_pez_prompts` function wraps `attention_control.cross_attention.
compute_token_mapping` and additionally returns the unmapped target
positions, which `LocalBlend` uses as its `target_token_indices`.
"""

from src.splice.align import align_pez_prompts

__all__ = ["align_pez_prompts"]
