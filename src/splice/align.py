"""Token-alignment between PEZ-1 source and PEZ-2 target prompts.

`align_pez_prompts` is the wrapper around the existing LCS alignment
in `attention_control.cross_attention.compute_token_mapping` that also
returns the unmapped target positions — the input `LocalBlend` needs
to build its mask.

Semantic alignment (a CLIP-embedding-based fallback for cases where LCS
is unreliable on PEZ-derived prompts) is documented in Appendix B of
the research proposal but not implemented here; LCS is the v1 default.
"""

from __future__ import annotations

from typing import Literal

from attention_control.cross_attention import compute_token_mapping


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

    mapping = compute_token_mapping(source_token_ids, target_token_ids)
    mapped_target = set(mapping.values())
    unmapped_target_indices = [
        i for i in range(len(target_token_ids)) if i not in mapped_target
    ]
    return mapping, unmapped_target_indices
