"""
Cross-attention controller for Prompt-to-Prompt (P2P) image editing.

Hooks into the cross-attention layers of a diffusion U-Net to intercept and
replace attention maps at runtime, enabling prompt-driven edits (word swap,
refinement, re-weighting) without retraining.

Usage
-----
Wrap the target model's attention forward method with `register_attention_control`,
specifying which layers to inject via `layer_indices` (or leave None to inject
all cross-attention layers):

    register_attention_control(unet, controller, layer_indices={6, 7, 8})

The `controller` must implement:

    controller(attn_weights, is_cross, place_in_unet) -> attn_weights

where:
    attn_weights   : Tensor[batch, heads, spatial, tokens]  — current attention maps
    is_cross       : bool  — True for cross-attention, False for self-attention
    place_in_unet  : str   — one of {"down", "mid", "up"}

Returns the (optionally modified) attention weights to use for this step.

P2P injection patterns
----------------------
- **Word swap**     : replace source attention maps with target maps for edited tokens
- **Refinement**    : blend source and target maps (source * (1-alpha) + target * alpha)
- **Re-weighting**  : scale attention maps for specific tokens by a scalar weight

Notes
-----
- Inject only cross-attention layers (is_cross=True) to preserve spatial structure.
- Restrict injection to decoder ("up") layers for style-preserving edits.
- Store and reset controller state between diffusion steps via `controller.reset()`.
"""