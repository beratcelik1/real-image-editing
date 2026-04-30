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
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Adaptive early stopping
# ---------------------------------------------------------------------------
#
# The optimization loop terminates before `num_steps` if the optimization
# variable's relative L2 movement over a sliding window falls below a
# threshold. Movement-based (rather than loss-based) because SDS-CFG loss
# is dominated by MSE noise from random t/ε sampling — even at convergence
# the loss value is nowhere near zero, and noise drowns out the convergence
# signal. The variable's L2 trajectory is the cleaner signal: once it
# stops moving meaningfully, we've converged.
#
# Hardcoded defaults rather than config knobs to keep the convergence
# criterion outside the user's tuning surface (cf. RESEARCH_PROPOSAL.md
# §3.1 on PEZ-2's weight_decay being similarly hardcoded).

_EARLY_STOP_MIN_STEPS = 100         # never stop before this many steps
_EARLY_STOP_CHECK_INTERVAL = 20     # check movement every k steps
_EARLY_STOP_REL_THRESHOLD = 1e-3    # relative L2 movement below this → stop


def _run_loop_with_early_stop(
    optimizer: torch.optim.Optimizer,
    loss_fn_eval: Callable[[], torch.Tensor],
    var: torch.Tensor,                 # the optimization variable, for movement tracking
    num_steps: int,
    loss_history_out: list[float] | None = None,
    progress_desc: str | None = None,
) -> None:
    """Run AdamW optimization with movement-based adaptive early stopping.

    `loss_fn_eval` is a 0-arg closure that computes and returns the loss
    (so callers can wrap soft_prompt = anchor + Δ or pass soft_prompt
    directly). `var` is the parameter whose relative L2 movement we
    monitor for the early-stop criterion.

    If ``loss_history_out`` is provided, the per-step total loss is
    appended to it (one ``.item()`` GPU→CPU sync per step; ~ms,
    negligible vs the SDS step cost). If ``progress_desc`` is provided,
    a tqdm bar is shown with the current loss in the postfix (refreshed
    every ``_EARLY_STOP_CHECK_INTERVAL`` steps).
    """
    pbar = tqdm(range(num_steps), desc=progress_desc) if progress_desc else None
    iter_range = pbar if pbar is not None else range(num_steps)
    prev_check = var.detach().clone()
    try:
        for step in iter_range:
            loss = loss_fn_eval()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss_history_out is not None:
                loss_history_out.append(loss.detach().item())

            if (step + 1) % _EARLY_STOP_CHECK_INTERVAL == 0:
                if pbar is not None:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                if step >= _EARLY_STOP_MIN_STEPS:
                    with torch.no_grad():
                        cur = var.detach()
                        rel_move = (cur - prev_check).norm() / (cur.norm() + 1e-8)
                    if rel_move.item() < _EARLY_STOP_REL_THRESHOLD:
                        msg = (
                            f"[pez_search] early stop at step {step+1}/{num_steps} "
                            f"(rel L2 movement {rel_move.item():.2e} < "
                            f"{_EARLY_STOP_REL_THRESHOLD:.0e} over last "
                            f"{_EARLY_STOP_CHECK_INTERVAL} steps)"
                        )
                        if pbar is not None:
                            pbar.write(msg)
                        else:
                            print(msg)
                        return
                    prev_check = cur.clone()
    finally:
        if pbar is not None:
            pbar.close()


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
    loss_history_out: list[float] | None = None,
    progress_desc: str | None = None,
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
        Maximum number of gradient steps. The optimization terminates
        early if the optimization variable's relative L2 movement
        over a sliding window falls below threshold (movement-based
        adaptive early stop — see ``_run_loop_with_early_stop``).
        Movement-based rather than loss-based because SDS-CFG loss is
        dominated by sampling noise. Defaults are hardcoded outside
        the user's tuning surface.
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
        _run_loop_with_early_stop(
            optimizer=optimizer,
            loss_fn_eval=lambda: loss_fn(anchor + delta),
            var=delta,
            num_steps=num_steps,
            loss_history_out=loss_history_out,
            progress_desc=progress_desc,
        )
        # Diagnostic: report how far the residual actually drifted. SDS
        # loss curves are dominated by t/ε sampling noise, so loss values
        # don't reveal whether optimization is moving the parameter; ||Δ||
        # does. ||Δ||/||anchor|| near 0 → wd is killing the SDS signal
        # (try lowering delta_weight_decay). Non-trivial ratio → optimization
        # moved meaningfully; visual A/B against the anchor confirms whether
        # the move was useful.
        with torch.no_grad():
            delta_norm = delta.norm().item()
            anchor_norm = anchor.norm().item()
            rel = delta_norm / (anchor_norm + 1e-8)
            tag = progress_desc or "pez_search (anchored)"
            print(
                f"[{tag}] final ||Δ|| = {delta_norm:.4f}, "
                f"||anchor|| = {anchor_norm:.4f}, "
                f"||Δ||/||anchor|| = {rel:.4f}"
            )
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
    initial_for_movement = soft_prompt.detach().clone()
    _run_loop_with_early_stop(
        optimizer=optimizer,
        loss_fn_eval=lambda: loss_fn(soft_prompt),
        var=soft_prompt,
        num_steps=num_steps,
        loss_history_out=loss_history_out,
        progress_desc=progress_desc,
    )
    # Diagnostic: how far did the soft prompt drift from its starting
    # point? Same rationale as the anchored branch above.
    with torch.no_grad():
        sp = soft_prompt.detach()
        move = (sp - initial_for_movement).norm().item()
        sp_norm = sp.norm().item()
        rel = move / (sp_norm + 1e-8)
        tag = progress_desc or "pez_search"
        print(
            f"[{tag}] final ||soft_prompt - init|| = {move:.4f}, "
            f"||soft_prompt|| = {sp_norm:.4f}, "
            f"relative = {rel:.4f}"
        )
    return soft_prompt.detach()
