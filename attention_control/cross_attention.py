"""
Cross-attention controller for Prompt-to-Prompt (P2P) image editing.

Hooks into the cross-attention layers of a diffusion U-Net to intercept and
replace attention maps at runtime, enabling prompt-driven edits (word swap,
re-weighting) without retraining.

Usage
-----
    from attention_control.cross_attention import (
        CrossAttentionController,
        compute_token_mapping_from_text,
        register_attention_control,
        unregister_attention_control,
    )

    # Build token mapping between source and target prompts
    mapping = compute_token_mapping_from_text(tokenizer, src_prompt, tgt_prompt)

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
(uncond_source, uncond_target, cond_source, cond_target).

Re-weighting has no batch layout requirement and works with any batch size.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import torch
from diffusers.models.attention_processor import Attention


# ---------------------------------------------------------------------------
# Token alignment utilities
# ---------------------------------------------------------------------------


def compute_token_mapping(
    source_ids: list[int],
    target_ids: list[int],
) -> dict[int, int]:
    """Align source and target token sequences via longest common subsequence.

    Returns ``{source_idx: target_idx}`` for every token that appears in both
    sequences at a matched position.  Unmatched tokens (insertions, deletions,
    substitutions) are omitted — their attention maps will *not* be copied from
    the source during word-swap injection.
    """
    matcher = SequenceMatcher(None, source_ids, target_ids, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for i in range(block.size):
            mapping[block.a + i] = block.b + i
    return mapping


def compute_token_mapping_from_text(
    tokenizer,
    source_prompt: str,
    target_prompt: str,
) -> dict[int, int]:
    """Tokenize two prompts and compute their token alignment.

    Convenience wrapper around :func:`compute_token_mapping` for text
    modalities that use a HuggingFace-style tokenizer.
    """
    source_ids = tokenizer.encode(source_prompt)
    target_ids = tokenizer.encode(target_prompt)
    return compute_token_mapping(source_ids, target_ids)


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
    ) -> None:
        self.num_steps = num_steps
        self.cross_replace_steps = cross_replace_steps
        self.token_mapping = token_mapping
        self.reweight_factors = reweight_factors
        self.reweight_steps = reweight_steps
        self.layer_indices = layer_indices

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
        """
        half = attn.shape[0] // 2
        if half == 0:
            return attn

        source = attn[:half]
        target = attn[half:].clone()

        for src_tok, tgt_tok in self.token_mapping.items():
            if src_tok < attn.shape[-1] and tgt_tok < attn.shape[-1]:
                target[:, :, :, tgt_tok] = source[:, :, :, src_tok]

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
