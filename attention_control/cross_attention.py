"""
Cross-attention controller for Prompt-to-Prompt (P2P) image editing.

Hooks into the cross-attention layers of a diffusion U-Net to intercept and
replace attention maps at runtime, enabling prompt-driven edits (word swap,
re-weighting) without retraining.

Usage
-----
    from attention_control.cross_attention import (
        CrossAttentionController,
        register_attention_control,
        unregister_attention_control,
    )
    from src.splice.align import align_pez_prompts

    # Build per-position token mapping between continuous PEZ embeddings.
    # Returns matched + unmapped position indices over the [N, D] sequence
    # (BOS/EOS offsets handled at the call site).
    matched, unmapped = align_pez_prompts(src_emb, tgt_emb, threshold=0.95)
    mapping = {i + 1: i + 1 for i in matched}  # +1 for BOS

    # Create controller
    controller = CrossAttentionController(
        num_steps=50,
        cross_replace_steps=0.8,
        token_mapping=mapping,
    )

    # Install on UNet
    register_attention_control(unet, controller)

    # Run denoising loop
    for t in timesteps:
        # ... UNet forward pass with [source, target] batch ...
        controller.step()

    # Cleanup
    unregister_attention_control(unet)

    # Access stored attention maps
    maps = controller.cross_attention_maps

Batch convention
----------------
Word-swap expects the batch dimension to be laid out as
``[source…, target…]``.  For CFG-free inference this means batch=2
(one source, one target); with classifier-free guidance it is batch=4
(``[uncond_src, cond_src, uncond_tgt, cond_tgt]``) — the src and tgt
halves are kept contiguous so ``attn[:half]`` / ``attn[half:]`` slice
the right rows for the injection.

Re-weighting has no batch layout requirement and works with any batch size.
"""

from __future__ import annotations

import torch
from diffusers.models.attention_processor import Attention


# ---------------------------------------------------------------------------
# Cross-attention controller
# ---------------------------------------------------------------------------


class CrossAttentionController:
    """Stateful controller for P2P cross-attention injection.

    Intercepts cross-attention maps during diffusion denoising and optionally
    applies word-swap and/or re-weighting edits.  Also stores every
    cross-attention map for later visualisation.

    The controller is **modality-agnostic**: it operates on raw attention
    weight tensors and a pre-computed token mapping.  Text-specific helpers
    (tokenisation, alignment) live in the free functions above.

    Parameters
    ----------
    num_steps : int
        Total number of diffusion denoising steps.
    cross_replace_steps : float
        Fraction of steps (from the start) during which word-swap injection
        is active.  0.8 means inject for the first 80% of steps.
    token_mapping : dict[int, int] | None
        ``{source_token_idx: target_token_idx}`` for word-swap.  Attention
        maps at matched target positions are replaced by the corresponding
        source maps.  ``None`` disables word-swap.
    reweight_factors : dict[int, float] | None
        ``{token_idx: scale}`` for re-weighting.  Attention at the given
        token positions is multiplied by *scale*, then renormalised.
        ``None`` disables re-weighting.
    reweight_steps : float
        Fraction of steps during which re-weighting is active.
    layer_indices : set[int] | None
        Indices of attention layers whose maps should be *edited*.
        ``None`` means all cross-attention layers.  All layers are still
        stored regardless of this setting.
    """

    def __init__(
        self,
        num_steps: int,
        cross_replace_steps: float = 0.8,
        token_mapping: dict[int, int] | None = None,
        reweight_factors: dict[int, float] | None = None,
        reweight_steps: float = 1.0,
        layer_indices: set[int] | None = None,
        local_blend=None,
    ) -> None:
        """
        Parameters
        ----------
        local_blend : LocalBlend | None
            Optional spatial-gating mechanism for additive edits. When
            provided, the controller (a) records cross-attention maps to
            the local-blend's target token positions during ``__call__``
            and (b) gates the word-swap injection by the resulting mask
            in ``_word_swap``. When ``None`` (default), the controller
            behaves identically to the original P2P implementation.
            See ``attention_control/local_blend.py``.
        """
        self.num_steps = num_steps
        self.cross_replace_steps = cross_replace_steps
        self.token_mapping = token_mapping
        self.reweight_factors = reweight_factors
        self.reweight_steps = reweight_steps
        self.layer_indices = layer_indices
        self.local_blend = local_blend

        self._cur_step: int = 0
        self._attention_store: dict[str, list[list[torch.Tensor]]] = {}
        self._step_buffer: dict[str, list[torch.Tensor]] = {}
        self._init_store()

    def _init_store(self) -> None:
        for loc in ("down", "mid", "up"):
            key = f"{loc}_cross"
            self._attention_store[key] = []
            self._step_buffer[key] = []

    # -- public interface --------------------------------------------------

    @property
    def cur_step(self) -> int:
        return self._cur_step

    @property
    def cross_attention_maps(self) -> dict[str, list[list[torch.Tensor]]]:
        """All stored cross-attention maps.

        Structure: ``{location_cross: [[layer_tensors_at_step_0], …]}``.
        Each tensor has shape ``[batch, heads, spatial, tokens]`` on CPU.
        """
        return self._attention_store

    def step(self) -> None:
        """Flush the current step's buffer and advance the step counter.

        Call once after each denoising step.
        """
        for key in self._step_buffer:
            if self._step_buffer[key]:
                self._attention_store[key].append(list(self._step_buffer[key]))
                self._step_buffer[key] = []
        if self.local_blend is not None:
            self.local_blend.step()
        self._cur_step += 1

    def reset(self) -> None:
        """Reset all state for a fresh denoising run."""
        self._cur_step = 0
        self._init_store()

    def __call__(
        self,
        attn_weights: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        layer_idx: int,
    ) -> torch.Tensor:
        """Intercept attention weights: store and optionally modify.

        Parameters
        ----------
        attn_weights : Tensor[batch, heads, spatial, tokens]
        is_cross : bool
            ``True`` for cross-attention, ``False`` for self-attention.
        place_in_unet : str
            One of ``"down"``, ``"mid"``, ``"up"``.
        layer_idx : int
            Global index of this attention layer in the U-Net.

        Returns
        -------
        Tensor — the (possibly modified) attention weights.
        """
        if not is_cross:
            return attn_weights

        # Store (always, regardless of layer filter).
        key = f"{place_in_unet}_cross"
        self._step_buffer[key].append(attn_weights.detach().cpu())

        # Record cross-attention for the local-blend mask (if present).
        # Mask used at step t comes from cross-attention recorded at
        # step t-1; LocalBlend handles the lag internally.
        if self.local_blend is not None:
            self.local_blend.record_cross_attention(attn_weights)

        # Check layer filter for editing.
        if self.layer_indices is not None and layer_idx not in self.layer_indices:
            return attn_weights

        # Word-swap injection.
        if (
            self.token_mapping is not None
            and self._cur_step < self.num_steps * self.cross_replace_steps
        ):
            attn_weights = self._word_swap(attn_weights)

        # Re-weighting injection.
        if (
            self.reweight_factors is not None
            and self._cur_step < self.num_steps * self.reweight_steps
        ):
            attn_weights = self._reweight(attn_weights)

        return attn_weights

    # -- edit operations ---------------------------------------------------

    def _word_swap(self, attn: torch.Tensor) -> torch.Tensor:
        """Replace target attention maps with source maps for mapped tokens.

        Batch layout: ``[source…, target…]`` — first half is source, second
        half is target.  Works for batch=2 (no CFG) and batch=4 (with CFG).

        If a ``LocalBlend`` is attached, the swap is gated spatially:
        outside the mask the source's attention column replaces the
        target's (full P2P injection); inside the mask the target's
        attention column is preserved (no injection, letting the new
        content render).
        """
        half = attn.shape[0] // 2
        if half == 0:
            return attn

        source = attn[:half]
        target = attn[half:].clone()

        # Optional spatial mask from LocalBlend (None at step 0, or always
        # None when no LocalBlend is attached). Mask is shape ``[spatial]``
        # in [0, 1] where 1 = "inside the edit region (don't inject)".
        spatial = attn.shape[2]
        mask_flat = None
        if self.local_blend is not None:
            mask_flat = self.local_blend.get_mask(spatial)

        for src_tok, tgt_tok in self.token_mapping.items():
            if src_tok < attn.shape[-1] and tgt_tok < attn.shape[-1]:
                if mask_flat is None:
                    target[:, :, :, tgt_tok] = source[:, :, :, src_tok]
                else:
                    # ``w == 0`` outside mask → use source (full inject).
                    # ``w == 1`` inside mask → keep target (no inject).
                    w = mask_flat.view(1, 1, spatial)
                    target[:, :, :, tgt_tok] = (
                        (1 - w) * source[:, :, :, src_tok]
                        + w * target[:, :, :, tgt_tok]
                    )

        return torch.cat([source, target], dim=0)

    def _reweight(self, attn: torch.Tensor) -> torch.Tensor:
        """Scale attention for specific tokens, then renormalise."""
        attn = attn.clone()
        for tok_idx, scale in self.reweight_factors.items():
            if tok_idx < attn.shape[-1]:
                attn[:, :, :, tok_idx] *= scale
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        return attn


# ---------------------------------------------------------------------------
# Diffusers-compatible attention processor
# ---------------------------------------------------------------------------


class P2PAttnProcessor:
    """Drop-in attention processor that routes maps through a controller.

    Follows the standard ``AttnProcessor`` protocol from *diffusers*.  The
    only addition is a call to the controller between computing attention
    probabilities and the value-weighted sum.
    """

    def __init__(
        self,
        controller: CrossAttentionController,
        place_in_unet: str,
        layer_idx: int,
    ) -> None:
        self.controller = controller
        self.place_in_unet = place_in_unet
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, seq_len, batch_size
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # ---- P2P injection ----
        heads = attn.heads
        bh, s, t = attention_probs.shape
        attn_4d = attention_probs.view(batch_size, heads, s, t)
        attn_4d = self.controller(
            attn_4d, is_cross, self.place_in_unet, self.layer_idx
        )
        attention_probs = attn_4d.reshape(bh, s, t)
        # -----------------------

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection + dropout.
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def register_attention_control(
    unet,
    controller: CrossAttentionController,
) -> None:
    """Install :class:`P2PAttnProcessor` on every attention layer of *unet*.

    The controller's ``layer_indices`` determines which layers are
    *edited*; all layers are instrumented for attention-map storage.
    """
    attn_procs: dict[str, P2PAttnProcessor] = {}

    for idx, name in enumerate(unet.attn_processors.keys()):
        if name.startswith("mid_block"):
            place = "mid"
        elif name.startswith("up_blocks"):
            place = "up"
        elif name.startswith("down_blocks"):
            place = "down"
        else:
            place = "mid"

        attn_procs[name] = P2PAttnProcessor(controller, place, layer_idx=idx)

    unet.set_attn_processor(attn_procs)


def unregister_attention_control(unet) -> None:
    """Restore the default attention processors on *unet*."""
    unet.set_default_attn_processor()
