"""Main editing pipeline: inversion -> attention extraction -> editing.

Three top-level entry points:

- ``invert_and_reconstruct``: legacy, used to verify DDIM inversion +
  null-text optimization on a source image with a manual prompt.
- ``edit_image``: the project's end-to-end editing entry point. Chains
  PEZ-1 (alternating R=2), PEZ-2 (instruction-conditioned), and
  :func:`run_p2p_edit`.
- ``run_p2p_edit``: the editing tail of ``edit_image``, decoupled so
  callers with token IDs from any source (BLIP captions, hand-written
  prompts, sweep scripts) can reuse the DDIM-invert + P2P + LocalBlend
  wiring without re-running PEZ.
"""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.utils import (
    decode_latent,
    encode_image,
    get_text_embeddings,
    get_uncond_embeddings,
    load_image,
    load_sd_components,
    get_device,
)
from src.inversion import ddim_inversion, null_text_optimization, reconstruct
from src.config import (
    EditConfig,
    LocalBlendConfig,
    Pez1Config,
    Pez2Config,
    load_edit,
    load_local_blend,
    load_pez_1,
    load_pez_2,
)
from src.pez.source_inversion import pez_invert_source
from src.pez.instruction_conditioned import pez_invert_with_instruction
from src.pez.losses import assemble_77_token_embedding, encode_through_text_model
from src.splice.align import align_pez_prompts
from attention_control.cross_attention import (
    CrossAttentionController,
    register_attention_control,
    unregister_attention_control,
)
from attention_control.local_blend import LocalBlend


def invert_and_reconstruct(
    image_path: str,
    prompt: str,
    device: torch.device | None = None,
    num_steps: int = 50,
    cfg_scale: float = 7.5,
    opt_steps: int = 10,
    lr: float = 1e-2,
    output_dir: str = "outputs",
) -> None:
    """Full inversion + reconstruction pipeline for verification.

    Loads a real image, inverts it with DDIM, optimizes null-text embeddings,
    reconstructs, and saves a side-by-side comparison with metrics.
    """
    if device is None:
        device = get_device()

    print(f"Device: {device}")
    print("Loading SD v2.1 components...")
    unet, vae, text_encoder, tokenizer, scheduler = load_sd_components(device)

    # Load and encode image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    latent = encode_image(image, vae, device)

    # Get text embeddings
    print(f"Prompt: {prompt}")
    text_emb = get_text_embeddings(prompt, tokenizer, text_encoder, device)
    uncond_emb = get_uncond_embeddings(tokenizer, text_encoder, device)

    # Stage 1: DDIM Inversion
    print(f"Running DDIM inversion ({num_steps} steps, CFG={cfg_scale})...")
    latent_T, latents_trajectory = ddim_inversion(
        latent,
        text_emb,
        uncond_emb,
        unet,
        scheduler,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
    )
    print(f"Inverted latent shape: {latent_T.shape}")

    # Stage 2: Null-text Optimization
    print(f"Running null-text optimization ({opt_steps} steps/timestep, lr={lr})...")
    null_embeddings = null_text_optimization(
        latents_trajectory,
        text_emb,
        uncond_emb,
        unet,
        scheduler,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        opt_steps=opt_steps,
        lr=lr,
    )
    print(f"Optimized {len(null_embeddings)} null embeddings")

    # Stage 3: Reconstruct
    print("Reconstructing...")
    recon_latent = reconstruct(
        latent_T,
        null_embeddings,
        text_emb,
        unet,
        scheduler,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
    )
    recon_image = decode_latent(recon_latent, vae)

    # Compute metrics
    orig_np = np.array(image).astype(np.float32)
    recon_np = np.array(recon_image).astype(np.float32)
    psnr = peak_signal_noise_ratio(orig_np, recon_np, data_range=255.0)
    ssim = structural_similarity(orig_np, recon_np, channel_axis=2, data_range=255.0)
    print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    # Save side-by-side
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    comparison = Image.new("RGB", (image.width * 2, image.height))
    comparison.paste(image, (0, 0))
    comparison.paste(recon_image, (image.width, 0))

    stem = Path(image_path).stem
    out_path = out_dir / f"{stem}_reconstruction.png"
    comparison.save(out_path)
    print(f"Saved comparison to {out_path}")

    # Also save reconstruction alone
    recon_image.save(out_dir / f"{stem}_recon.png")
    image.save(out_dir / f"{stem}_original.png")


def edit_image(
    image_path: str,
    instruction: str,
    output_dir: str = "outputs",
    pez_2_overrides: dict | None = None,
    edit_overrides: dict | None = None,
    sd_components: dict | None = None,
    clip_image_features: torch.Tensor | None = None,
    text_projection: torch.nn.Linear | None = None,
) -> Image.Image:
    """End-to-end real-image editing: image + instruction → edited image.

    Pipeline (RESEARCH_PROPOSAL.md §3.7):
      1. PEZ-1 (alternating R=2) → (source_token_ids, null_text)
      2. PEZ-2 (instruction-conditioned) → target_token_ids
      3. :func:`run_p2p_edit` → DDIM invert + P2P + LocalBlend → edited image

    Knob 1 / Knob 2 ablation (see §4.5):
      pez_2_overrides   — Knob 1 (lambda_instruction, gamma_anchor)
      edit_overrides    — Knob 2 (cross_replace_steps)
    Both override values are flat dicts whose keys match Pez2Config /
    EditConfig fields. Sub-config keys use dotted paths, e.g.
    ``"cross_attention.cross_replace_steps"``.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    instruction : str
        Natural-language edit instruction.
    output_dir : str
        Directory to save the edited image. Created if missing.
    pez_2_overrides : dict | None
        Knob 1 ablation knobs. Applied to the loaded Pez2Config.
    edit_overrides : dict | None
        Knob 2 ablation knobs. Applied to the loaded EditConfig.
    sd_components : dict | None
        Pre-loaded SD components ``{"unet", "vae", "text_encoder",
        "tokenizer", "scheduler"}``. If None, loaded internally.
    clip_image_features : Tensor[1, D] | None
        Pre-computed CLIP image embedding for vanilla-PEZ bootstrap in
        PEZ-1. Required (see docs/pez_conditional/DESIGN_CHOICES.md #5).
    text_projection : nn.Linear | None
        Optional CLIPModel.text_projection for joint-space comparison.
    """
    # 1. Load configs and apply overrides
    pez_1_config = load_pez_1()
    pez_2_config = load_pez_2()
    local_blend_config = load_local_blend()
    edit_config = load_edit()
    if pez_2_overrides:
        pez_2_config = _apply_overrides(pez_2_config, pez_2_overrides)
    if edit_overrides:
        edit_config = _apply_overrides_nested(edit_config, edit_overrides)

    # 2. Load SD components (or use provided)
    if sd_components is None:
        sd_components = _load_sd_components_dict(edit_config)

    image = load_image(image_path)

    # 3. PEZ-1: source-image inversion (alternating R=2)
    print("[edit_image] Step 1/3: PEZ-1 source inversion")
    source_embeddings, null_text = pez_invert_source(
        image=image,
        config=pez_1_config,
        sd_components=sd_components,
        clip_image_features=clip_image_features,
        text_projection=text_projection,
    )

    # 4. PEZ-2: instruction-conditioned target generation
    print("[edit_image] Step 2/3: PEZ-2 instruction-conditioned generation")
    target_embeddings = pez_invert_with_instruction(
        image=image,
        instruction=instruction,
        pez_1_embeddings=source_embeddings,
        null_text_per_timestep=null_text,
        config=pez_2_config,
        sd_components=sd_components,
    )

    # 5. P2P + LocalBlend edit (extracted as a public function so callers
    #    with embeddings from any source — BLIP-derived, hand-written,
    #    alternative inversions — can reuse the same wiring).
    print("[edit_image] Step 3/3: P2P + LocalBlend edit")
    edited = run_p2p_edit(
        image=image,
        source_embeddings=source_embeddings,
        target_embeddings=target_embeddings,
        sd_components=sd_components,
        edit_config=edit_config,
        local_blend_config=local_blend_config,
    )

    # 6. Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    out_path = out_dir / f"{stem}_edited.png"
    edited.save(out_path)
    print(f"[edit_image] Saved to {out_path}")
    return edited


def run_p2p_edit(
    image: Image.Image,
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    sd_components: dict,
    edit_config: EditConfig,
    local_blend_config: LocalBlendConfig,
) -> Image.Image:
    """Run a P2P + LocalBlend edit on *image* given source/target embeddings.

    This is the editing tail of :func:`edit_image`, decoupled so callers
    that already have continuous embeddings from any source (PEZ-1/2,
    alternative inversions, sweep scripts iterating over pre-computed
    PEZ-2 results, etc.) can reuse it without re-running PEZ-1 / PEZ-2.

    Pipeline:
      1. Run each [1, N, 768] embedding through CLIP's text encoder
         (wrapped with BOS/EOS/padding to 77 positions) to get the
         contextual hidden states SD's U-Net cross-attention consumes.
      2. DDIM-invert the source under its contextual encoding.
      3. Per-position cosine alignment between source and target
         embeddings → matched / unmapped position lists.
      4. Set up :class:`CrossAttentionController` + optional
         :class:`LocalBlend`.
      5. Run the editing denoising loop with a ``[source, target]`` batch.
      6. Decode the target half and return.

    Parameters
    ----------
    image
        Source PIL image.
    source_embeddings, target_embeddings : Tensor[1, N, 768] or [N, 768]
        Continuous CLIP-input embeddings produced by PEZ-1 / PEZ-2
        (or any compatible source). Apply any ``cross_replace_steps``
        overrides via :func:`_apply_overrides_nested` on ``edit_config``
        *before* calling this function.
    sd_components
        Dict ``{"unet", "vae", "text_encoder", "tokenizer", "scheduler"}``.
    edit_config
        Loaded :class:`EditConfig` with any overrides already applied.
        The cosine threshold used for alignment is
        ``edit_config.alignment_threshold``.
    local_blend_config
        Loaded :class:`LocalBlendConfig`. If ``.enabled`` is False or
        there are no unmapped target positions, LocalBlend is skipped
        and the controller behaves identically to the original P2P
        paper.

    Returns
    -------
    edited_image : PIL.Image.Image
        The edited target image.
    """
    if edit_config.mode != "replace":
        raise NotImplementedError(
            f"Edit mode {edit_config.mode!r} is not implemented in v1. "
            "Only 'replace' is currently supported. See "
            "RESEARCH_PROPOSAL.md §3.0 for the full mode taxonomy and "
            "§7 (Phases R5/R6) for the planned ADD and EXPLICIT_REPLACE "
            "implementations."
        )

    tokenizer = sd_components["tokenizer"]
    text_encoder = sd_components["text_encoder"]
    unet = sd_components["unet"]
    vae = sd_components["vae"]
    scheduler = sd_components["scheduler"]
    device = torch.device(edit_config.device)
    dtype = _str_to_dtype(edit_config.dtype)

    # Normalize shapes: caller may hand us [N, D] or [1, N, D].
    src = _ensure_batched(source_embeddings).to(device=device)
    tgt = _ensure_batched(target_embeddings).to(device=device)

    source_emb = _encode_continuous_prompt(src, text_encoder, tokenizer, dtype)
    target_emb = _encode_continuous_prompt(tgt, text_encoder, tokenizer, dtype)
    uncond_emb = get_uncond_embeddings(tokenizer, text_encoder, device)

    image_latent = encode_image(image, vae, device).to(dtype=dtype)
    z_T, _ = ddim_inversion(
        image_latent, source_emb, uncond_emb,
        unet, scheduler,
        num_steps=edit_config.ddim.num_steps,
        cfg_scale=edit_config.ddim.cfg_scale,
    )

    matched, unmapped_target = align_pez_prompts(
        src.squeeze(0), tgt.squeeze(0),
        threshold=edit_config.alignment_threshold,
        method=edit_config.alignment_method,
    )
    # Convert matched-position list to {src: tgt} mapping for the
    # CrossAttentionController. Warm-start gives same-position
    # correspondence, so the mapping is the identity over matched.
    # The position indices are 0-based over the [N, 768] sequence;
    # CLIP wraps them with BOS at 0, so add 1 to land on the right
    # 77-position contextual-encoding indices.
    mapping = {i + 1: i + 1 for i in matched}
    unmapped_in_77 = [i + 1 for i in unmapped_target]

    local_blend = None
    if local_blend_config.enabled and unmapped_in_77:
        local_blend = LocalBlend(
            target_token_indices=unmapped_in_77,
            threshold=local_blend_config.threshold,
            base_resolution=local_blend_config.base_resolution,
            dilate_iters=local_blend_config.dilate_iters,
        )

    layer_indices = (
        set(edit_config.cross_attention.layer_indices)
        if edit_config.cross_attention.layer_indices else None
    )
    cross_ctrl = CrossAttentionController(
        num_steps=edit_config.ddim.num_steps,
        cross_replace_steps=edit_config.cross_attention.cross_replace_steps,
        token_mapping=mapping,
        layer_indices=layer_indices,
        local_blend=local_blend,
    )
    register_attention_control(unet, cross_ctrl)
    edited_latent = _run_editing_loop(
        z_T=z_T,
        source_emb=source_emb,
        target_emb=target_emb,
        uncond_emb=uncond_emb,
        unet=unet,
        scheduler=scheduler,
        cross_ctrl=cross_ctrl,
        cfg_scale=edit_config.ddim.cfg_scale,
        num_steps=edit_config.ddim.num_steps,
    )
    unregister_attention_control(unet)

    return decode_latent(edited_latent[1:2], vae)


def _ensure_batched(emb: torch.Tensor) -> torch.Tensor:
    """Promote [N, D] to [1, N, D]; pass [1, N, D] through unchanged."""
    if emb.ndim == 2:
        return emb.unsqueeze(0)
    if emb.ndim == 3 and emb.shape[0] == 1:
        return emb
    raise ValueError(f"Expected [N, D] or [1, N, D]; got shape {tuple(emb.shape)}")


def _encode_continuous_prompt(
    soft_prompt: torch.Tensor,    # [1, N, D]
    text_encoder,
    tokenizer,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run a continuous PEZ embedding through CLIP's text model.

    Wraps the [1, N, D] soft prompt with BOS/EOS/padding to 77
    positions and runs the result through CLIP's text transformer
    (bypassing the embedding lookup since we already have the input
    embeddings). Returns [1, 77, D] contextual hidden states.
    """
    token_embedding = text_encoder.text_model.embeddings.token_embedding
    full_embeds, eos_pos, pos_ids, attn_mask = assemble_77_token_embedding(
        soft_prompt.to(dtype),
        token_embedding=token_embedding,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.no_grad():
        last_hidden, _ = encode_through_text_model(
            full_embeds, pos_ids, attn_mask, eos_pos, text_encoder,
        )
    return last_hidden  # [1, 77, D]


# ---------------------------------------------------------------------------
# Helpers for edit_image
# ---------------------------------------------------------------------------


def _run_editing_loop(
    z_T: torch.Tensor,
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    unet,
    scheduler,
    cross_ctrl,
    cfg_scale: float,
    num_steps: int,
) -> torch.Tensor:
    """Run the editing denoising loop with a [source, target] batch.

    Implements the core P2P loop: at each step run U-Net under
    (uncond, source, uncond, target) batch with CFG, then advance the
    cross-attention controller's step counter (which also advances any
    attached LocalBlend).
    """
    scheduler.set_timesteps(num_steps)
    # Stack into batch [source, target]
    latent = torch.cat([z_T, z_T.clone()], dim=0)

    for t in scheduler.timesteps:
        # CFG batch: [uncond_src, uncond_tgt, cond_src, cond_tgt]
        latent_in = torch.cat([latent, latent], dim=0)
        embed_in = torch.cat([
            uncond_emb, uncond_emb,         # uncond for both
            source_emb, target_emb,          # cond for source/target
        ], dim=0)
        with torch.no_grad():
            noise_pred = unet(latent_in, t, encoder_hidden_states=embed_in).sample
        # Split and apply CFG
        eps_uncond, eps_cond = noise_pred.chunk(2)
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        latent = scheduler.step(eps, t, latent).prev_sample

        # Advance the cross-attention controller (which also calls
        # local_blend.step() if attached).
        cross_ctrl.step()

    return latent  # [2, 4, H, W] — index 0=source, 1=target


def _apply_overrides(config_obj, overrides: dict):
    """Return a new config with flat overrides applied via dataclasses.replace."""
    from dataclasses import replace
    return replace(config_obj, **overrides)


def _apply_overrides_nested(config_obj, overrides: dict):
    """Apply potentially nested overrides like 'cross_attention.cross_replace_steps'."""
    from dataclasses import replace
    nested: dict = {}
    flat: dict = {}
    for k, v in overrides.items():
        if "." in k:
            parent, child = k.split(".", 1)
            nested.setdefault(parent, {})[child] = v
        else:
            flat[k] = v
    new_kwargs = {**flat}
    for parent, child_overrides in nested.items():
        sub = getattr(config_obj, parent)
        new_kwargs[parent] = replace(sub, **child_overrides)
    return replace(config_obj, **new_kwargs)


def _load_sd_components_dict(edit_config: EditConfig) -> dict:
    device = torch.device(edit_config.device)
    unet, vae, text_encoder, tokenizer, scheduler = load_sd_components(device)
    return {
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }


def _str_to_dtype(s: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[s]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.pipeline <image_path> <prompt>           # legacy reconstruct")
        print('  python -m src.pipeline edit <image_path> <instruction>  # editing')
        sys.exit(1)

    if sys.argv[1] == "edit":
        if len(sys.argv) < 4:
            print("Usage: python -m src.pipeline edit <image_path> <instruction>")
            sys.exit(1)
        edit_image(sys.argv[2], sys.argv[3])
    else:
        if len(sys.argv) < 3:
            print('Usage: python -m src.pipeline <image_path> <prompt>')
            sys.exit(1)
        invert_and_reconstruct(sys.argv[1], sys.argv[2])
