"""Continuous-PEZ optimization loop with a pluggable loss function.

Adapted from Wen et al. 2023's PEZ optimization machinery
(``optim_utils.optimize_prompt_loop``) with two changes:

1. Pluggable loss function (a callable ``soft_prompt -> scalar``).
2. **No straight-through vocabulary projection.** The soft prompt is
   the canonical output and is used directly by downstream code.

See RESEARCH_PROPOSAL.md §3.1 for the architectural rationale (and
``docs/pez_conditional/DESIGN_CHOICES.md`` for the decision to
re-implement vs. import from ``external/pez/``).

The ``nn_project`` helper is retained as a utility for inference-time
human-readable projection (snap-to-nearest-vocabulary for logging or
debugging) — it is *not* called inside the optimization loop.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def nn_project(
    curr_embeds: torch.Tensor,
    embedding_layer: torch.nn.Embedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap soft embeddings to the nearest CLIP vocabulary token.

    No straight-through gradient — this helper exists for inference-time
    human-readable projection (e.g., logging the projected token
    sequence for debugging). The optimization loop in :func:`pez_search`
    no longer uses this.

    Parameters
    ----------
    curr_embeds : Tensor[..., dim]
        Soft prompt embeddings (any leading shape).
    embedding_layer : nn.Embedding
        CLIP's token embedding layer.

    Returns
    -------
    projected : Tensor[..., dim]
        Embeddings of the nearest vocabulary tokens.
    nn_indices : Tensor[..., int64]
        The vocabulary token IDs at each position.
    """
    leading_shape = curr_embeds.shape[:-1]
    emb_dim = curr_embeds.shape[-1]

    flat = curr_embeds.reshape(-1, emb_dim)
    with torch.no_grad():
        flat_norm = F.normalize(flat.to(embedding_layer.weight.dtype), dim=-1)
        vocab_norm = F.normalize(embedding_layer.weight, dim=-1)
        sim = flat_norm @ vocab_norm.T
        nn_indices = sim.argmax(dim=-1)
        projected = embedding_layer(nn_indices)

    projected = projected.reshape(*leading_shape, emb_dim).to(curr_embeds.dtype)
    nn_indices = nn_indices.reshape(*leading_shape)
    return projected, nn_indices


def pez_search(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    token_embedding: torch.nn.Embedding,
    prompt_length: int,
    num_steps: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    initial_soft_prompt: torch.Tensor | None = None,
    anchor_to: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the continuous-PEZ optimization loop with a caller-supplied loss.

    Parameters
    ----------
    loss_fn
        Callable taking a tensor of shape ``[1, prompt_length, dim]``
        (the soft prompt) and returning a scalar loss tensor. The loss
        can use any external machinery (CLIP, U-Net, etc.) — gradients
        propagate back to the soft prompt directly.
    token_embedding
        CLIP's ``nn.Embedding`` for the token vocabulary. Used only for
        random warm-start initialization (when both ``anchor_to`` and
        ``initial_soft_prompt`` are None); not consulted inside the
        optimization loop.
    prompt_length
        N — the number of soft-prompt positions to optimize.
    num_steps
        Number of gradient steps.
    learning_rate, weight_decay
        AdamW hyperparameters. ``weight_decay`` is applied to whichever
        tensor is the optimization variable — soft_prompt directly
        (anchored or unanchored warm start), or Δ in the residual
        parameterization (when ``anchor_to`` is provided).
    seed
        Random seed; controls the random init when no warm-start.
    device
        Where to run the optimization.
    initial_soft_prompt : Tensor[1, prompt_length, dim] or None
        Plain warm start. If provided, soft_prompt is initialized at
        this tensor and AdamW operates on soft_prompt directly. Used
        by PEZ-2 (whose anchor lives in the loss as L_anchor).
    anchor_to : Tensor[1, prompt_length, dim] or None
        Residual-parameterization anchor. If provided, optimize
        Δ (initialized at 0); soft_prompt is ``anchor_to + Δ`` at
        every step. AdamW's weight_decay decays Δ → 0, which decays
        soft_prompt → anchor_to. Used by PEZ-1's SDS rounds (round
        1+) to make the previous round's output a stable equilibrium
        of the current round's optimization. See RESEARCH_PROPOSAL.md
        §3.1 "Residual parameterization for SDS rounds."

        ``anchor_to`` and ``initial_soft_prompt`` are mutually
        exclusive — providing both raises ValueError.

    Returns
    -------
    final_soft_prompt : Tensor[1, prompt_length, dim]
        The final soft prompt — the canonical PEZ output. Use directly
        as input to CLIP's text encoder; do not project to discrete
        vocabulary unless you specifically need a human-readable form
        for logging.
    """
    if initial_soft_prompt is not None and anchor_to is not None:
        raise ValueError(
            "anchor_to and initial_soft_prompt are mutually exclusive. "
            "Use anchor_to for the residual-parameterization warm start "
            "(decays soft_prompt toward anchor_to via weight_decay on Δ); "
            "use initial_soft_prompt for a plain warm start (weight_decay "
            "decays soft_prompt toward origin). See RESEARCH_PROPOSAL.md §3.1."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    expected_shape = (1, prompt_length, token_embedding.weight.shape[1])

    if anchor_to is not None:
        # Residual parameterization: optimize Δ (init at 0).
        # soft_prompt = anchor_to + Δ at every step.
        if anchor_to.shape != expected_shape:
            raise ValueError(
                f"anchor_to shape {tuple(anchor_to.shape)} does not match "
                f"{expected_shape}"
            )
        anchor = anchor_to.detach().clone().to(device=device, dtype=torch.float32)
        delta = torch.zeros_like(anchor, requires_grad=True)
        optimizer = torch.optim.AdamW(
            [delta], lr=learning_rate, weight_decay=weight_decay
        )
        for _ in range(num_steps):
            loss = loss_fn(anchor + delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return (anchor + delta).detach()

    # Non-anchored path: optimize soft_prompt directly.
    if initial_soft_prompt is None:
        vocab_size = token_embedding.weight.shape[0]
        random_ids = torch.randint(
            0, vocab_size, (1, prompt_length), device=device
        )
        soft_prompt = token_embedding(random_ids).detach().clone()
    else:
        if initial_soft_prompt.shape != expected_shape:
            raise ValueError(
                f"initial_soft_prompt shape "
                f"{tuple(initial_soft_prompt.shape)} does not match "
                f"{expected_shape}"
            )
        soft_prompt = initial_soft_prompt.detach().clone()

    # Optimize in float32 for numerical stability — Adam interacts
    # badly with fp16 grads.
    soft_prompt = soft_prompt.to(device=device, dtype=torch.float32)
    soft_prompt.requires_grad = True

    optimizer = torch.optim.AdamW(
        [soft_prompt], lr=learning_rate, weight_decay=weight_decay
    )

    for _ in range(num_steps):
        loss = loss_fn(soft_prompt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return soft_prompt.detach()
