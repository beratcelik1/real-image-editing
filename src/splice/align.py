"""Token-alignment between PEZ-1 source and PEZ-2 target prompts.

`align_pez_prompts` returns both the LCS mapping (compatible with
``CrossAttentionController.token_mapping``) and the unmapped target
positions — the latter is what ``LocalBlend`` needs as its
``target_token_indices``.

The LCS logic mirrors the algorithm in
``attention_control/cross_attention.py:compute_token_mapping`` but is
inlined here so this module can be imported without pulling in torch.

Semantic alignment (a CLIP-embedding-based fallback for cases where LCS
is unreliable on PEZ-derived prompts) is documented in Appendix B of
the research proposal but not implemented here; LCS is the v1 default.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Literal


def _compute_token_mapping_lcs(
    source_ids: list[int],
    target_ids: list[int],
) -> dict[int, int]:
    """Mirror of ``attention_control.cross_attention.compute_token_mapping``.

    Inlined so this module is torch-free.
    """
    matcher = SequenceMatcher(None, source_ids, target_ids, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for i in range(block.size):
            mapping[block.a + i] = block.b + i
    return mapping


def align_pez_prompts(
    source_token_ids: list[int],
    target_token_ids: list[int],
    method: Literal["lcs", "semantic"] = "lcs",
) -> tuple[dict[int, int], list[int]]:
    """Align two PEZ-derived prompts and return both the mapping and
    the unmapped target indices.

    Parameters
    ----------
    source_token_ids
        Discrete CLIP-vocabulary token IDs for the source prompt
        (typically PEZ-1's output).
    target_token_ids
        Discrete token IDs for the target prompt (typically PEZ-2's).
    method
        "lcs": longest common subsequence over token IDs (default,
        fast, sufficient for warm-started PEZ-2 outputs).
        "semantic": CLIP-embedding-based bipartite matching. Not
        implemented in v1 — see RESEARCH_PROPOSAL.md Appendix B.

    Returns
    -------
    mapping : dict[int, int]
        ``{source_pos: target_pos}`` for matched tokens. Compatible
        with ``CrossAttentionController.token_mapping``.
    unmapped_target_indices : list[int]
        Target positions with no source counterpart. These are the
        edit positions and feed ``LocalBlend.target_token_indices``.
    """
    if method == "semantic":
        raise NotImplementedError(
            "Semantic alignment is not implemented in v1. "
            "See RESEARCH_PROPOSAL.md Appendix B for the spec; "
            "fall back to method='lcs'."
        )

    mapping = _compute_token_mapping_lcs(source_token_ids, target_token_ids)
    mapped_target = set(mapping.values())
    unmapped_target_indices = [
        i for i in range(len(target_token_ids)) if i not in mapped_target
    ]
    return mapping, unmapped_target_indices
