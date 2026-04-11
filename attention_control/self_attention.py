"""
Self-attention controller for Plug-and-Play (PnP) image editing.

Injects source self-attention maps into the target denoising process to
preserve spatial layout and structure while allowing prompt-driven edits.

Usage (standalone)
------------------
    from attention_control.self_attention import (
        SelfAttentionController,
        register_self_attention_control,
        unregister_attention_control,
    )

    controller = SelfAttentionController(
        num_steps=50,
        self_replace_steps=0.5,
    )

    register_self_attention_control(unet, controller)

    # Run denoising loop with [source, target] batch
    for t in timesteps:
        # ... UNet forward pass ...
        controller.step()

    unregister_attention_control(unet)

Usage (combined with cross-attention)
-------------------------------------
    from attention_control.cross_attention import CrossAttentionController
    from attention_control.self_attention import (
        SelfAttentionController,
        register_combined_control,
    )

    cross_ctrl = CrossAttentionController(num_steps=50, ...)
    self_ctrl  = SelfAttentionController(num_steps=50, ...)

    register_combined_control(unet, cross_ctrl, self_ctrl)

    for t in timesteps:
        # ... UNet forward pass ...
        cross_ctrl.step()
        self_ctrl.step()

    unregister_attention_control(unet)

Batch convention
----------------
Same as ``cross_attention.py``: the batch dimension is laid out as
``[source…, target…]``.  Target self-attention maps are replaced with the
source maps so the target preserves the source's spatial structure.

Memory note
-----------
Self-attention maps have shape ``[batch, heads, spatial, spatial]`` and can
be very large at high resolutions (e.g. 4096 x 4096 at 64 x 64 spatial).
Map storage is **off by default**; enable with ``store_maps=True`` if you
need visualisation, but be aware of memory cost.
"""

from __future__ import annotations

from typing import Protocol

import torch
from diffusers.models.attention_processor import Attention


# ---------------------------------------------------------------------------
# Controller protocol (shared interface for composability)
# ---------------------------------------------------------------------------


class AttentionController(Protocol):
    """Minimal interface that any attention controller must satisfy.

    Both :class:`~cross_attention.CrossAttentionController` and
    :class:`SelfAttentionController` implement this protocol, which is what
    :class:`CombinedAttnProcessor` relies on for chaining.
    """

    def __call__(
        self,
        attn_weights: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        layer_idx: int,
    ) -> torch.Tensor: ...

    def step(self) -> None: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Self-attention controller
# ---------------------------------------------------------------------------


class SelfAttentionController:
    """Stateful controller for PnP self-attention injection.

    Intercepts self-attention maps during diffusion denoising and replaces
    the target's maps with the source's maps so that spatial layout is
    preserved across an edit.

    The controller is **modality-agnostic**: it operates on raw attention
    weight tensors with a ``[source…, target…]`` batch convention and has
    no text-specific logic.

    Parameters
    ----------
    num_steps : int
        Total number of diffusion denoising steps.
    self_replace_steps : float
        Fraction of steps (from the start) during which self-attention
        injection is active.  0.5 means inject for the first 50 % of steps.
        Lower values allow the target more freedom to deviate from the
        source layout.
    layer_indices : set[int] | None
        Indices of attention layers whose maps should be injected.
        ``None`` means all self-attention layers.
    store_maps : bool
        If ``True``, store every self-attention map on CPU for later
        visualisation.  **Off by default** because self-attention maps are
        ``[batch, heads, spatial, spatial]`` and can consume significant
        memory at high resolutions.
    """

    def __init__(
        self,
        num_steps: int,
        self_replace_steps: float = 0.5,
        layer_indices: set[int] | None = None,
        store_maps: bool = False,
    ) -> None:
        self.num_steps = num_steps
        self.self_replace_steps = self_replace_steps
        self.layer_indices = layer_indices
        self.store_maps = store_maps

        self._cur_step: int = 0
        self._attention_store: dict[str, list[list[torch.Tensor]]] = {}
        self._step_buffer: dict[str, list[torch.Tensor]] = {}
        self._init_store()

    def _init_store(self) -> None:
        for loc in ("down", "mid", "up"):
            key = f"{loc}_self"
            self._attention_store[key] = []
            self._step_buffer[key] = []

    # -- public interface --------------------------------------------------

    @property
    def cur_step(self) -> int:
        return self._cur_step

    @property
    def self_attention_maps(self) -> dict[str, list[list[torch.Tensor]]]:
        """All stored self-attention maps (empty when ``store_maps=False``).

        Structure: ``{location_self: [[layer_tensors_at_step_0], …]}``.
        Each tensor has shape ``[batch, heads, spatial, spatial]`` on CPU.
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
        """Intercept attention weights: optionally store and inject.

        Parameters
        ----------
        attn_weights : Tensor[batch, heads, spatial, spatial]
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
        # Only handle self-attention; pass cross-attention through unchanged.
        if is_cross:
            return attn_weights

        # Optionally store.
        if self.store_maps:
            key = f"{place_in_unet}_self"
            self._step_buffer[key].append(attn_weights.detach().cpu())

        # Check layer filter for injection.
        if self.layer_indices is not None and layer_idx not in self.layer_indices:
            return attn_weights

        # Self-attention injection.
        if self._cur_step < self.num_steps * self.self_replace_steps:
            attn_weights = self._inject(attn_weights)

        return attn_weights

    # -- injection ---------------------------------------------------------

    def _inject(self, attn: torch.Tensor) -> torch.Tensor:
        """Replace target self-attention maps with source maps.

        Batch layout: ``[source…, target…]``.  The source half is kept
        unchanged; the target half receives a copy of the source maps.
        """
        half = attn.shape[0] // 2
        if half == 0:
            return attn

        source = attn[:half]
        return torch.cat([source, source.clone()], dim=0)


# ---------------------------------------------------------------------------
# Diffusers-compatible attention processors
# ---------------------------------------------------------------------------


def _attention_forward(
    attn_module: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    temb: torch.Tensor | None,
    controllers: list[AttentionController],
    place_in_unet: str,
    layer_idx: int,
) -> torch.Tensor:
    """Shared attention forward pass used by all processor classes.

    Computes standard scaled dot-product attention, passes the attention
    probabilities through every *controller* in order, then produces the
    final output.  Factored out so ``PnPAttnProcessor`` and
    ``CombinedAttnProcessor`` stay DRY.
    """
    is_cross = encoder_hidden_states is not None
    residual = hidden_states

    if attn_module.spatial_norm is not None:
        hidden_states = attn_module.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(
            batch_size, channel, height * width
        ).transpose(1, 2)

    batch_size, seq_len, _ = hidden_states.shape

    if attention_mask is not None:
        attention_mask = attn_module.prepare_attention_mask(
            attention_mask, seq_len, batch_size
        )

    if attn_module.group_norm is not None:
        hidden_states = attn_module.group_norm(
            hidden_states.transpose(1, 2)
        ).transpose(1, 2)

    query = attn_module.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn_module.norm_cross:
        encoder_hidden_states = attn_module.norm_encoder_hidden_states(
            encoder_hidden_states
        )

    key = attn_module.to_k(encoder_hidden_states)
    value = attn_module.to_v(encoder_hidden_states)

    query = attn_module.head_to_batch_dim(query)
    key = attn_module.head_to_batch_dim(key)
    value = attn_module.head_to_batch_dim(value)

    attention_probs = attn_module.get_attention_scores(query, key, attention_mask)

    # ---- controller injection ----
    heads = attn_module.heads
    bh, s, t = attention_probs.shape
    attn_4d = attention_probs.view(batch_size, heads, s, t)
    for ctrl in controllers:
        attn_4d = ctrl(attn_4d, is_cross, place_in_unet, layer_idx)
    attention_probs = attn_4d.reshape(bh, s, t)
    # ------------------------------

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn_module.batch_to_head_dim(hidden_states)

    # Linear projection + dropout.
    hidden_states = attn_module.to_out[0](hidden_states)
    hidden_states = attn_module.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch_size, channel, height, width
        )

    if attn_module.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn_module.rescale_output_factor

    return hidden_states


class PnPAttnProcessor:
    """Drop-in attention processor for PnP self-attention injection.

    Structurally identical to
    :class:`~cross_attention.P2PAttnProcessor` — only the default
    controller type differs.  Accepts any object that satisfies the
    :class:`AttentionController` protocol.
    """

    def __init__(
        self,
        controller: AttentionController,
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
        return _attention_forward(
            attn, hidden_states, encoder_hidden_states, attention_mask,
            temb, [self.controller], self.place_in_unet, self.layer_idx,
        )


class CombinedAttnProcessor:
    """Attention processor that chains multiple controllers.

    Each controller is called in order on the attention probabilities.
    Controllers that only handle one attention type (e.g.
    :class:`~cross_attention.CrossAttentionController` ignores
    self-attention, :class:`SelfAttentionController` ignores
    cross-attention) compose naturally — they simply pass through
    attention types they do not handle.

    Example
    -------
    ::

        from attention_control.cross_attention import CrossAttentionController
        from attention_control.self_attention import (
            SelfAttentionController,
            register_combined_control,
        )

        cross_ctrl = CrossAttentionController(num_steps=50, token_mapping=mapping)
        self_ctrl  = SelfAttentionController(num_steps=50)
        register_combined_control(unet, cross_ctrl, self_ctrl)
    """

    def __init__(
        self,
        controllers: list[AttentionController],
        place_in_unet: str,
        layer_idx: int,
    ) -> None:
        self.controllers = controllers
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
        return _attention_forward(
            attn, hidden_states, encoder_hidden_states, attention_mask,
            temb, self.controllers, self.place_in_unet, self.layer_idx,
        )


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def _resolve_place(name: str) -> str:
    """Determine U-Net location from an attention processor key name."""
    if name.startswith("mid_block"):
        return "mid"
    if name.startswith("up_blocks"):
        return "up"
    if name.startswith("down_blocks"):
        return "down"
    return "mid"


def register_self_attention_control(
    unet,
    controller: SelfAttentionController,
) -> None:
    """Install :class:`PnPAttnProcessor` on every attention layer of *unet*.

    The controller's ``layer_indices`` determines which layers are
    injected; all self-attention layers are observed when
    ``store_maps=True``.
    """
    attn_procs: dict[str, PnPAttnProcessor] = {}
    for idx, name in enumerate(unet.attn_processors.keys()):
        attn_procs[name] = PnPAttnProcessor(
            controller, _resolve_place(name), layer_idx=idx,
        )
    unet.set_attn_processor(attn_procs)


def register_combined_control(
    unet,
    *controllers: AttentionController,
) -> None:
    """Install :class:`CombinedAttnProcessor` that chains *controllers*.

    Each controller is called in the order given.  Controllers that only
    handle one attention type compose naturally (cross-attention controllers
    pass self-attention through, and vice versa).
    """
    ctrl_list = list(controllers)
    attn_procs: dict[str, CombinedAttnProcessor] = {}
    for idx, name in enumerate(unet.attn_processors.keys()):
        attn_procs[name] = CombinedAttnProcessor(
            ctrl_list, _resolve_place(name), layer_idx=idx,
        )
    unet.set_attn_processor(attn_procs)


def unregister_attention_control(unet) -> None:
    """Restore the default attention processors on *unet*."""
    unet.set_default_attn_processor()
