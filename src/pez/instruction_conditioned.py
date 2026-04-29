"""PEZ-2 — instruction-conditioned target embedding generation.

Three-term loss (RESEARCH_PROPOSAL.md §3.2):
  L = L_source + lambda_instruction * L_instruction + gamma_anchor * L_anchor

- L_source: SDS-CFG with the source's frozen null-text (same form as
  PEZ-1's loss).
- L_instruction: text-text CLIP cosine between the prompt and the
  instruction text.
- L_anchor: L2 in soft-prompt space to PEZ-1's continuous embeddings.

Output is a continuous Tensor[1, prompt_length, 768], same shape as
PEZ-1's output. No vocabulary projection.
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
    sample_sds_timestep_idx,
    sds_cfg_loss_from_encoded,
)
from src.pez.search import pez_search


def _hash_pez2(
    image: Image.Image,
    instruction: str,
    pez_1_embeddings: torch.Tensor,
    null_text_per_timestep: list[torch.Tensor],
    config: Pez2Config,
) -> str:
    """Cache key for PEZ-2: every input that affects the result.

    L_source uses null_text_per_timestep, so changes to the null-text
    that don't propagate through pez_1_embeddings (e.g. tweaking PEZ-1's
    null-text-optim opt_steps/lr while leaving the embeddings unchanged)
    must invalidate the cache. Hashes the concatenated tensor bytes.
    """
    h = hashlib.sha256()
    h.update(image.tobytes())
    h.update(image.size[0].to_bytes(4, "little"))
    h.update(image.size[1].to_bytes(4, "little"))
    h.update(instruction.encode("utf-8"))
    h.update(pez_1_embeddings.detach().cpu().contiguous().numpy().tobytes())
    null_text_stacked = torch.stack(
        [nt.detach().cpu().contiguous() for nt in null_text_per_timestep],
        dim=0,
    )
    h.update(null_text_stacked.numpy().tobytes())
    cfg_str = (
        f"{config.source_loss_type}|cfg={config.cfg_scale}|"
        f"ts_sampling={config.timestep_sampling}|"
        f"lam={config.lambda_instruction}|gam={config.gamma_anchor}|"
        f"warm={config.warm_start}|"
        f"steps={config.num_steps}|lr={config.learning_rate}|"
        f"seed={config.seed}"
    )
    h.update(cfg_str.encode("utf-8"))
    return h.hexdigest()[:16]


def pez_invert_with_instruction(
    image: Image.Image,
    instruction: str,
    pez_1_embeddings: torch.Tensor,
    null_text_per_timestep: list[torch.Tensor],
    config: Pez2Config,
    sd_components: dict,
) -> torch.Tensor:
    """Run PEZ-2 with the three-term joint loss.

    Parameters
    ----------
    image
        Source image (used for the SDS term's image_latent).
    instruction
        User's natural-language edit instruction.
    pez_1_embeddings : Tensor[1, prompt_length, 768]
        PEZ-1's continuous source embeddings (the warm-start source).
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
    target_embeddings : Tensor[1, prompt_length, 768]
        Continuous target embeddings in CLIP-input space. Feed to
        CLIP's text encoder for downstream P2P editing.
    """
    # Cache check
    image_hash = _hash_pez2(
        image, instruction, pez_1_embeddings, null_text_per_timestep, config,
    )
    cache_file = Path(config.cache_dir) / f"{image_hash}.pt"
    if config.use_cache and cache_file.exists():
        cached = torch.load(cache_file, map_location="cpu")
        return cached["target_embeddings"]

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

    # Stack null-text for t-sampled lookup. Move to device explicitly
    # — null_text_per_timestep may have come from torch.load(map_location='cpu')
    # so .to(device) here prevents a device-mismatch crash inside U-Net
    # forward (Bug #3 fix).
    null_text_stacked = torch.stack(
        [nt.detach().to(device=device) for nt in null_text_per_timestep], dim=0,
    )  # [T, 1, 77, D]
    T = null_text_stacked.shape[0]
    # Cache scheduler.timesteps so we can map t_idx → actual timestep
    # value for U-Net forward (Bug #1 fix — keeps null-text and t aligned).
    scheduler.set_timesteps(T)
    scheduler_timesteps = scheduler.timesteps.to(device=device)

    # Warm-start: identity copy of PEZ-1's continuous embeddings.
    # No vocabulary lookup — we're already in CLIP-input embedding space.
    if config.warm_start:
        soft_prompt_init = pez_1_embeddings.detach().clone().to(device=device)
        if soft_prompt_init.ndim == 2:
            soft_prompt_init = soft_prompt_init.unsqueeze(0)
        prompt_length = soft_prompt_init.shape[1]
    else:
        soft_prompt_init = None
        prompt_length = pez_1_embeddings.shape[-2]

    # Capture the init for the L_anchor term
    soft_prompt_init_anchor = (
        soft_prompt_init.detach().clone() if soft_prompt_init is not None else None
    )

    # Build the joint loss closure
    def _joint_loss_fn(soft_prompt: torch.Tensor) -> torch.Tensor:
        # Encode the soft prompt through CLIP once and reuse the
        # contextual hidden state for L_source (cross-attn input to
        # U-Net) and the pooled output for L_instruction. The prior
        # implementation encoded the soft prompt twice per gradient
        # step (Bug #7).
        full_embeds, eos_pos, pos_ids, attn_mask = assemble_77_token_embedding(
            soft_prompt,
            token_embedding=token_embedding,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        full_embeds = full_embeds.to(dtype=unet.dtype)
        last_hidden, prompt_pooled = encode_through_text_model(
            full_embeds, pos_ids, attn_mask, eos_pos, text_encoder,
        )

        # L_source: SDS-CFG. Sample t_idx (centralized in
        # losses.sample_sds_timestep_idx — supports uniform /
        # uniform_truncated / importance) and look up BOTH the
        # matching null-text AND the matching training-timestep value
        # so they stay coherent (Bug #1). Sampling t independently
        # inside the loss would use null-text optimized for one
        # timestep with U-Net at a different timestep, producing
        # garbage eps_uncond.
        t_idx = sample_sds_timestep_idx(
            timestep_sampling=config.timestep_sampling,
            T=T,
            scheduler_timesteps=scheduler_timesteps,
            num_train_timesteps=scheduler.config.num_train_timesteps,
            device=device,
        )
        null_text_for_t = null_text_stacked[t_idx.item()]
        t = scheduler_timesteps[t_idx.item()].view(1).to(device=device, dtype=torch.long)

        l_source = sds_cfg_loss_from_encoded(
            last_hidden_state=last_hidden,
            image_latent=image_latent,
            null_text_embedding=null_text_for_t,
            t=t,
            cfg_scale=config.cfg_scale,
            unet=unet,
            scheduler=scheduler,
        )

        # L_instruction: text-text CLIP cosine
        prompt_pooled_n = F.normalize(prompt_pooled, dim=-1)
        instr_pooled_n = F.normalize(instr_pooled.to(prompt_pooled.dtype), dim=-1)
        l_instr = -(prompt_pooled_n * instr_pooled_n).sum(dim=-1).mean()

        # L_anchor: L2 to warm-start init
        if soft_prompt_init_anchor is not None:
            l_anchor = (
                (soft_prompt - soft_prompt_init_anchor.to(soft_prompt.dtype)) ** 2
            ).sum()
        else:
            l_anchor = torch.tensor(0.0, device=device, dtype=soft_prompt.dtype)

        return (
            l_source
            + config.lambda_instruction * l_instr
            + config.gamma_anchor * l_anchor
        )

    print(f"[PEZ-2] instruction-conditioned optimization: {instruction!r}")
    target_embeddings = pez_search(
        loss_fn=_joint_loss_fn,
        token_embedding=token_embedding,
        prompt_length=prompt_length,
        num_steps=config.num_steps,
        learning_rate=config.learning_rate,
        # PEZ-2's anchor is the L_anchor term in the joint loss above
        # (||soft_prompt - soft_prompt_init||² with γ = gamma_anchor).
        # AdamW.weight_decay is hardcoded to 0.0 — NOT a config knob —
        # because any positive value would silently pull soft_prompt
        # toward CLIP-space-origin and conflict with L_anchor pulling
        # toward soft_prompt_init. That contamination would distort the
        # (λ, γ) ablation grid in §4.2 of the research proposal.
        # See RESEARCH_PROPOSAL.md §3.1 "Residual parameterization for
        # SDS rounds" for the symmetric PEZ-1 fix and the rationale for
        # keeping these as two distinct mechanisms (residual parameter-
        # ization vs explicit L_anchor loss term) at the two stages.
        weight_decay=0.0,
        seed=config.seed,
        device=device,
        initial_soft_prompt=soft_prompt_init,
    )  # [1, N, D]

    if config.use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"target_embeddings": target_embeddings}, cache_file)

    return target_embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_to_dtype(s: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[s]


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
