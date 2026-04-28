"""Core PEZ optimization loop with a pluggable loss function.

Adapted from Wen et al. 2023 ``optim_utils.optimize_prompt_loop`` but
loss-pluggable and using only PyTorch primitives (no
``sentence_transformers``). The straight-through projection step is
the critical adapted piece; everything else is a standard
gradient-descent loop.

See docs/pez_conditional/DESIGN_CHOICES.md for the decision to
re-implement vs. import from ``external/pez/``.
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
    """Straight-through projection of soft embeddings to nearest CLIP vocab.

    Forward: each row of ``curr_embeds`` is replaced by the embedding of
    its nearest vocabulary token (cosine similarity).
    Backward: identity (gradient flows to ``curr_embeds`` as if no
    projection happened).

    Parameters
    ----------
    curr_embeds : Tensor[batch, seq_len, dim]
        The current soft prompt embeddings.
    embedding_layer : nn.Embedding
        CLIP's token embedding layer (e.g.,
        ``text_encoder.text_model.embeddings.token_embedding``).
        Weight shape: ``[vocab_size, dim]``.

    Returns
    -------
    projected : Tensor[batch, seq_len, dim]
        Embeddings of the nearest vocab tokens, with a straight-through
        gradient route back to ``curr_embeds``.
    nn_indices : Tensor[batch, seq_len], int64
        The vocabulary token IDs at each position.
    """
    bsz, seq_len, emb_dim = curr_embeds.shape

    flat = curr_embeds.reshape(-1, emb_dim)
    with torch.no_grad():
        flat_norm = F.normalize(flat.to(embedding_layer.weight.dtype), dim=-1)
        vocab_norm = F.normalize(embedding_layer.weight, dim=-1)
        sim = flat_norm @ vocab_norm.T  # [bsz*seq_len, vocab_size]
        nn_indices = sim.argmax(dim=-1)  # [bsz*seq_len]
        projected = embedding_layer(nn_indices)  # [bsz*seq_len, emb_dim]

    projected = projected.reshape(bsz, seq_len, emb_dim).to(curr_embeds.dtype)
    nn_indices = nn_indices.reshape(bsz, seq_len)

    # Straight-through: forward = projected (numerically), backward = identity.
    return projected.detach() + curr_embeds - curr_embeds.detach(), nn_indices


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
    projection_every: int = 1,
) -> tuple[list[int], torch.Tensor]:
    """Run the PEZ optimization loop with a caller-supplied loss.

    Parameters
    ----------
    loss_fn
        Callable taking a tensor of shape ``[1, prompt_length, dim]``
        (the projected soft prompt with straight-through gradients) and
        returning a scalar loss tensor. The loss can use any external
        machinery (CLIP, U-Net, etc.) — gradients propagate through
        the projection back to the soft prompt.
    token_embedding
        CLIP's ``nn.Embedding`` for the token vocabulary. Used both
        for the projection step and for warm-start initialization.
    prompt_length
        N — the number of vocabulary tokens to optimize.
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
    projection_every
        Project to vocabulary every k steps. The straight-through
        gradient still flows; lower values stay closer to discrete.
        Default 1 (every step).

    Returns
    -------
    token_ids : list[int]
        The discovered discrete CLIP vocabulary token IDs (length N).
    final_soft_prompt : Tensor[1, prompt_length, dim]
        The final soft prompt (useful for warm-starting other PEZ runs).
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

    last_indices: torch.Tensor | None = None
    for step in range(num_steps):
        if step % projection_every == 0:
            projected, last_indices = nn_project(soft_prompt, token_embedding)
        else:
            # Use the last-known projection result for the forward pass
            # but keep the soft prompt gradient route active.
            projected = (
                soft_prompt.detach() + soft_prompt - soft_prompt.detach()
            )

        loss = loss_fn(projected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final readout: project once more to get clean discrete tokens.
    with torch.no_grad():
        _, final_indices = nn_project(soft_prompt, token_embedding)

    return final_indices.squeeze(0).tolist(), soft_prompt.detach()
