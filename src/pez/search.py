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
        random warm-start initialization (when
        ``initial_soft_prompt is None``); not consulted inside the
        optimization loop.
    prompt_length
        N — the number of soft-prompt positions to optimize.
    num_steps
        Number of gradient steps.
    learning_rate, weight_decay
        AdamW hyperparameters.
    seed
        Random seed; controls the random init when warm-start is None.
    device
        Where to run the optimization.
    initial_soft_prompt : Tensor[1, prompt_length, dim] or None
        Warm-start. If None, initialize from random vocabulary tokens.

    Returns
    -------
    final_soft_prompt : Tensor[1, prompt_length, dim]
        The final soft prompt — the canonical PEZ output. Use directly
        as input to CLIP's text encoder; do not project to discrete
        vocabulary unless you specifically need a human-readable form
        for logging.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if initial_soft_prompt is None:
        vocab_size = token_embedding.weight.shape[0]
        random_ids = torch.randint(
            0, vocab_size, (1, prompt_length), device=device
        )
        soft_prompt = token_embedding(random_ids).detach().clone()
    else:
        if initial_soft_prompt.shape != (1, prompt_length, token_embedding.weight.shape[1]):
            raise ValueError(
                f"initial_soft_prompt shape "
                f"{tuple(initial_soft_prompt.shape)} does not match "
                f"(1, {prompt_length}, {token_embedding.weight.shape[1]})"
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
