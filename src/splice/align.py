"""Per-position cosine alignment between PEZ-1 source and PEZ-2 target embeddings.

Continuous PEZ produces same-length [N, 768] embedding sequences for
source and target with per-position correspondence by warm-start
construction. Alignment reduces to a per-position cosine-similarity
threshold: positions whose target stayed close to source are matched
(P2P injects K/V); positions that drifted are unmapped (P2P leaves
alone, and they drive ``LocalBlend.target_token_indices``).

See RESEARCH_PROPOSAL.md §3.3 and Appendix B for the design rationale
and alternatives.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def align_pez_prompts(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    threshold: float = 0.95,
    method: Literal["cosine_threshold"] = "cosine_threshold",
) -> tuple[list[int], list[int]]:
    """Per-position cosine alignment between two PEZ-derived prompts.

    Parameters
    ----------
    source_embeddings : Tensor[..., N, 768]
        PEZ-1's source embeddings. The leading batch dimension (if any)
        is squeezed.
    target_embeddings : Tensor[..., N, 768]
        PEZ-2's target embeddings, same shape as ``source_embeddings``.
    threshold : float
        Cosine-similarity cutoff τ. Positions with cos_sim ≥ threshold
        are matched; positions below are unmapped.
    method : "cosine_threshold"
        Only supported method in v1. The legacy ``"lcs"`` and unbuilt
        ``"semantic"`` methods are removed (see RESEARCH_PROPOSAL.md
        Appendix B for alternatives if cosine threshold proves
        insufficient empirically).

    Returns
    -------
    matched_indices : list[int]
        Positions where ``cos_sim(src[i], tgt[i]) >= threshold``.
        These positions receive P2P K/V injection.
    unmapped_target_indices : list[int]
        Positions where cosine is below the threshold. These drive
        ``LocalBlend.target_token_indices``.
    """
    if method != "cosine_threshold":
        raise NotImplementedError(
            f"Alignment method {method!r} is not implemented in v1. "
            "Only 'cosine_threshold' is supported. See "
            "RESEARCH_PROPOSAL.md Appendix B for alternatives if "
            "the per-position cosine threshold proves insufficient."
        )

    src = source_embeddings.reshape(-1, source_embeddings.shape[-1])
    tgt = target_embeddings.reshape(-1, target_embeddings.shape[-1])
    if src.shape != tgt.shape:
        raise ValueError(
            f"source/target embedding shapes mismatch: "
            f"{tuple(src.shape)} vs {tuple(tgt.shape)}. "
            "Continuous PEZ should produce same-length sequences via "
            "warm-start; check that PEZ-2 used pez_1_embeddings.clone() "
            "as its initial soft prompt."
        )

    sims = F.cosine_similarity(src.float(), tgt.float(), dim=-1)
    matched: list[int] = []
    unmapped: list[int] = []
    for i, s in enumerate(sims.tolist()):
        if s >= threshold:
            matched.append(i)
        else:
            unmapped.append(i)
    return matched, unmapped
