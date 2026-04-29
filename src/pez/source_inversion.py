"""PEZ-1 source-image inversion via alternating R=2 optimization.

See RESEARCH_PROPOSAL.md §3.1 for the algorithm. This module wires
together vanilla PEZ (CLIP loss bootstrap), null-text optimization
(existing in ``src/inversion.py``), and SDS-CFG-PEZ refinement
(``src/pez/losses.py:sds_cfg_loss``) into a single function.

Caching: per-image hash plus per-round results, so partial progress
survives crashes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.config import Pez1Config
from src.inversion import ddim_inversion, null_text_optimization
from src.pez.losses import (
    assemble_77_token_embedding,
    clip_similarity_loss,
    encode_through_text_model,
    sds_cfg_loss,
)
from src.pez.search import pez_search
from src.utils import encode_image, get_uncond_embeddings, load_sd_components


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _hash_image_and_config(image: Image.Image, config: Pez1Config) -> str:
    """Hash key for caching: image bytes + every config field that
    affects the optimization result.

    Includes (Bug #4 fix): weight_decay (round-0 reg), delta_weight_decay
    (round-1+ residual anchor — the new key knob from §3.1), learning_rate,
    timestep_sampling. Tuning any of these changes the output, so they
    must invalidate the cache.
    """
    h = hashlib.sha256()
    h.update(image.tobytes())
    h.update(image.size[0].to_bytes(4, "little"))
    h.update(image.size[1].to_bytes(4, "little"))
    config_str = (
        f"{config.loss_type}|cfg={config.cfg_scale}|"
        f"ts_sampling={config.timestep_sampling}|"
        f"N={config.prompt_length}|steps={config.num_steps}|"
        f"lr={config.learning_rate}|"
        f"wd={config.weight_decay}|dwd={config.delta_weight_decay}|"
        f"R={config.num_rounds}|seed={config.seed}|"
        f"clip={config.clip_model}"
    )
    h.update(config_str.encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_path(image_hash: str, config: Pez1Config) -> Path:
    return Path(config.cache_dir) / f"{image_hash}.pt"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pez_invert_source(
    image: Image.Image,
    config: Pez1Config,
    sd_components: dict | None = None,
    clip_image_features: torch.Tensor | None = None,
    text_projection: torch.nn.Linear | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run PEZ-1 alternating R=N pipeline on a source image.

    Returns
    -------
    source_embeddings : Tensor[1, prompt_length, 768]
        Continuous CLIP-input embeddings — the canonical PEZ-1 output.
        Feed directly to CLIP's text encoder via
        ``losses.assemble_77_token_embedding`` + ``encode_through_text_model``
        for downstream use.
    null_text_per_timestep : list[Tensor]
        Optimized null-text embeddings, one per denoising timestep,
        each shape ``[1, 77, 768]``. To be used at editing time and
        reused as input to PEZ-2.

    Parameters
    ----------
    image
        Source PIL image.
    config
        Loaded ``Pez1Config`` (typically ``load_pez_1()``).
    sd_components
        Optional pre-loaded SD components dict
        ``{"unet", "vae", "text_encoder", "tokenizer", "scheduler"}``.
        If None, calls ``load_sd_components()`` internally.
    clip_image_features
        Optional pre-computed CLIP image embedding ``[1, D]`` for the
        bootstrap clip_similarity_loss. If None, will need to be supplied
        externally — vanilla PEZ requires a CLIP image-text similarity
        target. (We don't load a CLIP model here to avoid a heavy
        dependency; the caller is responsible.)

        See docs/pez_conditional/DESIGN_CHOICES.md for the rationale.
    text_projection
        Optional ``CLIPModel.text_projection`` for the joint-space
        comparison. If None, comparison is in the text encoder's
        native output space.
    """
    # 1. Cache check
    image_hash = _hash_image_and_config(image, config)
    cache_file = _cache_path(image_hash, config)
    if config.use_cache and cache_file.exists():
        cached = torch.load(cache_file, map_location="cpu")
        return cached["source_embeddings"], cached["null_text"]

    # 2. Set up models
    if sd_components is None:
        sd_components = _load_sd_components_dict(config)
    device = torch.device(config.device)
    dtype = _str_to_dtype(config.dtype)
    unet = sd_components["unet"]
    vae = sd_components["vae"]
    text_encoder = sd_components["text_encoder"]
    tokenizer = sd_components["tokenizer"]
    scheduler = sd_components["scheduler"]
    token_embedding = text_encoder.text_model.embeddings.token_embedding

    # 3. Encode the image to its VAE latent (used in SDS loss)
    image_latent = encode_image(image, vae, device).to(dtype=dtype)

    # 4. Round 0 — vanilla PEZ bootstrap (CLIP-cosine loss)
    if clip_image_features is None:
        raise ValueError(
            "pez_invert_source requires `clip_image_features` for the "
            "vanilla-PEZ bootstrap step. Compute the CLIP image embedding "
            "of the source image externally and pass it in. "
            "See docs/pez_conditional/DESIGN_CHOICES.md for why."
        )

    def _clip_loss_fn(projected: torch.Tensor) -> torch.Tensor:
        return clip_similarity_loss(
            projected,
            target_image_features=clip_image_features,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_projection=text_projection,
        )

    print("[PEZ-1 Round 0] vanilla PEZ bootstrap (CLIP-cosine loss)")
    soft_prompt = pez_search(
        loss_fn=_clip_loss_fn,
        token_embedding=token_embedding,
        prompt_length=config.prompt_length,
        num_steps=config.num_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=config.seed,
        device=device,
        initial_soft_prompt=None,
    )  # [1, N, D]

    # 5. Round 1..R — alternating null-text + SDS-PEZ refinement
    null_text: list[torch.Tensor] | None = None
    uncond_emb = get_uncond_embeddings(tokenizer, text_encoder, device)
    for round_idx in range(1, config.num_rounds + 1):
        print(f"[PEZ-1 Round {round_idx}] null-text + SDS-PEZ refinement")

        # 5a. Encode the current soft prompt through CLIP's text model
        # to get [1, 77, D] contextual hidden states for ddim_inversion
        # and null-text optimization. (No tokenizer.decode round-trip;
        # the soft prompt is already in CLIP-input embedding space.)
        text_emb = _encode_soft_prompt(
            soft_prompt, text_encoder, tokenizer, dtype,
        )

        _, latents_traj = ddim_inversion(
            image_latent,
            text_emb,
            uncond_emb,
            unet,
            scheduler,
            num_steps=50,                 # standard
            cfg_scale=config.cfg_scale,
        )

        # 5b. Null-text optimization with the existing implementation.
        null_text = null_text_optimization(
            latents_traj,
            text_emb,
            uncond_emb,
            unet,
            scheduler,
            num_steps=50,
            cfg_scale=config.cfg_scale,
            opt_steps=10,
            lr=1e-2,
        )

        # 5c. SDS-PEZ refinement with frozen null-text.
        # Move null-text tensors to the optimization device explicitly
        # (they may have been loaded from CPU on cache hit — see Bug #3).
        # Stack along a leading T dim so t_idx-sampling is a clean lookup.
        null_text_stacked = torch.stack(
            [nt.detach().to(device=device) for nt in null_text], dim=0,
        )  # [T, 1, 77, D]
        # scheduler.timesteps gives the actual training-timestep value
        # at each denoising-step index; we need this so the wrapper can
        # pass the matching `t` into sds_cfg_loss alongside null_text[t_idx].
        scheduler.set_timesteps(null_text_stacked.shape[0])
        scheduler_timesteps = scheduler.timesteps.to(device=device)

        def _sds_loss_fn(soft_prompt_in: torch.Tensor) -> torch.Tensor:
            # Sample t and the matching null-text together; pass both.
            return _sds_loss_with_t_sampled_null_text(
                soft_prompt_in,
                image_latent=image_latent,
                null_text_per_timestep=null_text_stacked,
                scheduler_timesteps=scheduler_timesteps,
                cfg_scale=config.cfg_scale,
                unet=unet,
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                timestep_sampling=config.timestep_sampling,
            )

        # Residual parameterization for SDS rounds (proposal §3.1):
        # anchor_to = previous round's output, optimize Δ (init at 0).
        # weight_decay = delta_weight_decay (anchor strength on Δ).
        soft_prompt = pez_search(
            loss_fn=_sds_loss_fn,
            token_embedding=token_embedding,
            prompt_length=config.prompt_length,
            num_steps=config.num_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.delta_weight_decay,
            seed=config.seed + round_idx,
            device=device,
            anchor_to=soft_prompt,
        )

    # 6. Cache and return
    if config.use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"source_embeddings": soft_prompt, "null_text": null_text},
            cache_file,
        )

    return soft_prompt, null_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_to_dtype(s: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[s]


def _load_sd_components_dict(config: Pez1Config) -> dict:
    """Wrap ``src.utils.load_sd_components`` in a dict by name."""
    device = torch.device(config.device)
    unet, vae, text_encoder, tokenizer, scheduler = load_sd_components(device)
    return {
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }


def _encode_soft_prompt(
    soft_prompt: torch.Tensor,    # [1, N, D]
    text_encoder,
    tokenizer,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run a continuous PEZ soft prompt through CLIP's text model.

    Wraps the [1, N, D] soft prompt with BOS/EOS/padding to reach 77
    positions, then runs the wrapped sequence through CLIP's text
    transformer (bypassing the embedding lookup since we already have
    the input embeddings). Returns the [1, 77, D] contextual hidden
    states that downstream code (ddim_inversion, null-text opt) feeds
    to SD's U-Net cross-attention.
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


def _sds_loss_with_t_sampled_null_text(
    projected: torch.Tensor,
    image_latent: torch.Tensor,
    null_text_per_timestep: torch.Tensor,    # [T, 1, 77, D]
    scheduler_timesteps: torch.Tensor,        # [T] long, denoising-step → actual timestep
    cfg_scale: float,
    unet,
    scheduler,
    text_encoder,
    tokenizer,
    timestep_sampling: str,
) -> torch.Tensor:
    """SDS loss with null-text and U-Net timestep sampled coherently.

    Critical: ``null_text_per_timestep[t_idx]`` and the timestep used
    for the U-Net forward pass must be the SAME timestep — null-text
    was optimized for a specific denoising-step index. This wrapper
    samples ``t_idx ∈ [0, T)``, looks up null-text, AND derives the
    actual training-timestep value ``t = scheduler_timesteps[t_idx]``,
    passing both to ``sds_cfg_loss``. Sampling them independently
    (the prior bug — see Bug #1 in the audit commit) breaks the
    consistency: U-Net sees x_t at one noise level with null-text
    optimized for a different noise level, producing garbage
    eps_uncond.
    """
    device = image_latent.device
    T = null_text_per_timestep.shape[0]

    if timestep_sampling == "uniform":
        t_idx = torch.randint(0, T, (1,), device=device, dtype=torch.long)
    else:  # "importance" — bias toward mid-timesteps
        u = torch.rand(1, device=device)
        t_idx = ((1 - torch.sqrt(1 - u)) * T).long().clamp_(0, T - 1)

    null_text_for_t = null_text_per_timestep[t_idx.item()]      # [1, 77, D]
    t = scheduler_timesteps[t_idx.item()].view(1).to(device=device, dtype=torch.long)

    return sds_cfg_loss(
        projected,
        image_latent=image_latent,
        null_text_embedding=null_text_for_t,
        t=t,
        cfg_scale=cfg_scale,
        unet=unet,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )
