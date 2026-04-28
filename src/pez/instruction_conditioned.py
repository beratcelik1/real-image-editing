"""PEZ-2 — instruction-conditioned target prompt generation.

Three-term loss (RESEARCH_PROPOSAL.md §3.2):
  L = L_source + lambda_instruction * L_instruction + gamma_anchor * L_anchor

- L_source: SDS-CFG with the source's frozen null-text (same form as
  PEZ-1's loss).
- L_instruction: text-text CLIP cosine between the prompt and the
  instruction text.
- L_anchor: L2 in soft-prompt space to PEZ-1's vocabulary embeddings.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import Pez2Config
from src.pez.losses import (
    assemble_77_token_embedding,
    encode_through_text_model,
    sds_cfg_loss,
)
from src.pez.search import pez_search


def _hash_pez2(
    image: Image.Image,
    instruction: str,
    pez_1_token_ids: list[int],
    config: Pez2Config,
) -> str:
    h = hashlib.sha256()
    h.update(image.tobytes())
    h.update(image.size[0].to_bytes(4, "little"))
    h.update(image.size[1].to_bytes(4, "little"))
    h.update(instruction.encode("utf-8"))
    h.update(",".join(str(i) for i in pez_1_token_ids).encode("utf-8"))
    cfg_str = (
        f"{config.source_loss_type}|cfg={config.cfg_scale}|"
        f"lam={config.lambda_instruction}|gam={config.gamma_anchor}|"
        f"warm={config.warm_start}|steps={config.num_steps}|"
        f"seed={config.seed}|clip={config.clip_model}"
    )
    h.update(cfg_str.encode("utf-8"))
    return h.hexdigest()[:16]


def pez_invert_with_instruction(
    image: Image.Image,
    instruction: str,
    pez_1_token_ids: list[int],
    null_text_per_timestep: list[torch.Tensor],
    config: Pez2Config,
    sd_components: dict,
) -> list[int]:
    """Run PEZ-2 with the three-term joint loss.

    Parameters
    ----------
    image
        Source image (used for the SDS term's image_latent).
    instruction
        User's natural-language edit instruction.
    pez_1_token_ids
        PEZ-1's discrete output tokens (the warm-start source).
    null_text_per_timestep
        PEZ-1's optimized null-text (one per denoising timestep).
        Frozen during PEZ-2.
    config
        Loaded ``Pez2Config``.
    sd_components
        Dict ``{"unet", "vae", "text_encoder", "tokenizer", "scheduler"}``.
        Required (no internal load). Passed by the caller for cost
        amortization across multiple PEZ-2 runs on the same image.

    Returns
    -------
    target_token_ids : list[int]
        Discrete target prompt token IDs.
    """
    # Cache check
    image_hash = _hash_pez2(image, instruction, pez_1_token_ids, config)
    cache_file = Path(config.cache_dir) / f"{image_hash}.pt"
    if config.use_cache and cache_file.exists():
        cached = torch.load(cache_file, map_location="cpu")
        return cached["target_token_ids"]

    device = torch.device(config.device)
    dtype = _str_to_dtype(config.dtype)

    unet = sd_components["unet"]
    vae = sd_components["vae"]
    text_encoder = sd_components["text_encoder"]
    tokenizer = sd_components["tokenizer"]
    scheduler = sd_components["scheduler"]
    token_embedding = text_encoder.text_model.embeddings.token_embedding

    # Encode source image latent (for the SDS term)
    from src.utils import encode_image
    image_latent = encode_image(image, vae, device).to(dtype=dtype)

    # Pre-compute the instruction's pooled CLIP text encoding
    # (used in L_instruction).
    instr_pooled = _encode_text_pooled(instruction, tokenizer, text_encoder, device)

    # Stack null-text for t-sampled lookup
    null_text_stacked = torch.stack(null_text_per_timestep, dim=0)  # [T, 1, 77, D]
    T = null_text_stacked.shape[0]

    # Warm-start from PEZ-1's vocab embeddings
    if config.warm_start:
        ids_tensor = torch.tensor([pez_1_token_ids], device=device)
        soft_prompt_init = token_embedding(ids_tensor).detach().clone()
    else:
        soft_prompt_init = None

    # Capture the init for the L_anchor term
    soft_prompt_init_anchor = (
        soft_prompt_init.detach().clone() if soft_prompt_init is not None else None
    )

    # Build the joint loss closure
    def _joint_loss_fn(projected: torch.Tensor) -> torch.Tensor:
        # L_source: SDS-CFG with t-sampled null-text
        if config.timestep_sampling == "uniform":
            t_idx = torch.randint(0, T, (1,), device=device, dtype=torch.long)
        else:
            u = torch.rand(1, device=device)
            t_idx = ((1 - torch.sqrt(1 - u)) * T).long().clamp_(0, T - 1)
        null_text_for_t = null_text_stacked[t_idx.item()]

        l_source = sds_cfg_loss(
            projected,
            image_latent=image_latent,
            null_text_embedding=null_text_for_t,
            cfg_scale=config.cfg_scale,
            unet=unet,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            timestep_sampling=config.timestep_sampling,
        )

        # L_instruction: text-text CLIP cosine
        full_embeds, eos_pos, pos_ids, attn_mask = assemble_77_token_embedding(
            projected,
            token_embedding=token_embedding,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        _, prompt_pooled = encode_through_text_model(
            full_embeds, pos_ids, attn_mask, eos_pos, text_encoder,
        )
        prompt_pooled_n = F.normalize(prompt_pooled, dim=-1)
        instr_pooled_n = F.normalize(instr_pooled.to(prompt_pooled.dtype), dim=-1)
        l_instr = -(prompt_pooled_n * instr_pooled_n).sum(dim=-1).mean()

        # L_anchor: L2 to warm-start init
        if soft_prompt_init_anchor is not None:
            l_anchor = (
                (projected - soft_prompt_init_anchor.to(projected.dtype)) ** 2
            ).sum()
        else:
            l_anchor = torch.tensor(0.0, device=device, dtype=projected.dtype)

        return (
            l_source
            + config.lambda_instruction * l_instr
            + config.gamma_anchor * l_anchor
        )

    print(f"[PEZ-2] instruction-conditioned optimization: {instruction!r}")
    target_ids, _ = pez_search(
        loss_fn=_joint_loss_fn,
        token_embedding=token_embedding,
        prompt_length=len(pez_1_token_ids),
        num_steps=config.num_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.1,                  # standard
        seed=config.seed,
        device=device,
        initial_soft_prompt=soft_prompt_init,
        projection_every=config.projection_every,
    )

    if config.use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"target_token_ids": target_ids}, cache_file)

    return target_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_to_dtype(s: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[s]


def _encode_text_pooled(
    text: str, tokenizer, text_encoder, device,
) -> torch.Tensor:
    """Return the pooled CLIP text encoding for a string."""
    tokens = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = text_encoder(tokens.input_ids.to(device))
    # CLIPTextModel returns BaseModelOutputWithPooling; pooler_output is
    # the [CLS]/EOS-position pooled vector.
    return out.pooler_output  # [1, D]
