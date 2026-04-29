"""PEZ loss functions: vanilla CLIP-cosine and CFG-aware SDS.

Both losses take a continuous soft-prompt tensor of shape
``[1, prompt_length, dim]`` and return a scalar loss. Gradient flows
back to the soft prompt directly (continuous-PEZ — no straight-through
vocabulary projection). See RESEARCH_PROPOSAL.md §3.1 for the loss
formulations.

The soft prompt is assembled into a 77-position CLIP-input sequence
via ``assemble_77_token_embedding`` (BOS/EOS/padding wrapping); see
``docs/pez_conditional/DESIGN_CHOICES.md`` for embedding-assembly
choices.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers — assembling a 77-token CLIP encoder input from a soft prompt
# ---------------------------------------------------------------------------


def assemble_77_token_embedding(
    soft_prompt: torch.Tensor,         # [1, N, D] continuous embedding
    token_embedding: torch.nn.Embedding,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    max_length: int = 77,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Pad ``soft_prompt`` with BOS/EOS/padding to reach ``max_length``.

    The output structure mirrors what CLIP's tokenizer would produce
    for a length-N prompt: ``[BOS, t_1, ..., t_N, EOS, PAD, ..., PAD]``.

    Returns
    -------
    full_embeds : Tensor[1, max_length, D]
        The full 77-position input embedding sequence.
    eos_position : int
        Index of the EOS token (used for pooling).
    position_ids : Tensor[1, max_length]
        Standard ``arange`` for CLIP's position embeddings.
    attention_mask : Tensor[1, max_length]
        All-ones (CLIP attends to padding too — that's how it was trained).
    """
    bsz, prompt_len, dim = soft_prompt.shape
    if bsz != 1:
        raise NotImplementedError("Multi-batch not supported in PEZ loss assembly")
    device = soft_prompt.device

    pad_len = max_length - prompt_len - 2  # subtract BOS + EOS
    if pad_len < 0:
        raise ValueError(
            f"prompt_length={prompt_len} is too long: "
            f"need ≤ {max_length - 2} after reserving BOS+EOS"
        )

    with torch.no_grad():
        bos_emb = token_embedding(
            torch.tensor([bos_token_id], device=device)
        ).to(soft_prompt.dtype)  # [1, D]
        eos_emb = token_embedding(
            torch.tensor([eos_token_id], device=device)
        ).to(soft_prompt.dtype)  # [1, D]
        pad_emb = token_embedding(
            torch.tensor([pad_token_id], device=device)
        ).to(soft_prompt.dtype)  # [1, D]

    # Expand to [1, _, D] for concat
    bos_expanded = bos_emb.unsqueeze(0)                  # [1, 1, D]
    eos_expanded = eos_emb.unsqueeze(0)                  # [1, 1, D]
    pad_expanded = pad_emb.unsqueeze(0).expand(1, pad_len, dim)  # [1, pad_len, D]

    full_embeds = torch.cat(
        [bos_expanded, soft_prompt, eos_expanded, pad_expanded], dim=1
    )

    eos_position = 1 + prompt_len  # [BOS, t_1, ..., t_N, EOS, ...]

    position_ids = torch.arange(max_length, device=device).unsqueeze(0)
    attention_mask = torch.ones(1, max_length, device=device, dtype=torch.long)

    return full_embeds, eos_position, position_ids, attention_mask


def encode_through_text_model(
    full_embeds: torch.Tensor,         # [1, 77, D]
    position_ids: torch.Tensor,        # [1, 77]
    attention_mask: torch.Tensor,      # [1, 77]
    eos_position: int,
    text_encoder,                      # transformers CLIPTextModel
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a custom input embedding through CLIP's text transformer.

    Bypasses the embedding lookup (since we already have embeddings
    from the soft prompt's projection) but otherwise replicates
    ``CLIPTextTransformer.forward``.

    Returns
    -------
    last_hidden_state : Tensor[1, 77, D]
        The 77×768 contextual encoding (what SD's cross-attention uses).
    pooled_output : Tensor[1, D]
        The pooled embedding at ``eos_position`` (what CLIP cosine
        similarity uses).
    """
    text_model = text_encoder.text_model

    # Add positional embeddings (we already have token embeddings via
    # the soft-prompt projection).
    position_embeddings = text_model.embeddings.position_embedding(position_ids)
    hidden_states = full_embeds + position_embeddings

    # Causal attention mask. CLIP's text encoder is causal.
    bsz, seq_len, _ = hidden_states.shape
    causal_attention_mask = _build_causal_attention_mask(
        bsz, seq_len, hidden_states.dtype, hidden_states.device
    )

    # Encoder
    encoder_outputs = text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=None,                  # all-ones; redundant
        causal_attention_mask=causal_attention_mask,
    )
    last_hidden_state = encoder_outputs.last_hidden_state
    last_hidden_state = text_model.final_layer_norm(last_hidden_state)

    # Pooled output: at the EOS position. Standard CLIP convention.
    pooled_output = last_hidden_state[:, eos_position, :]

    return last_hidden_state, pooled_output


def _build_causal_attention_mask(bsz, seq_len, dtype, device):
    """Replicate transformers' internal causal mask shape used by CLIP."""
    mask = torch.empty(bsz, 1, seq_len, seq_len, device=device, dtype=dtype)
    mask.fill_(torch.finfo(dtype).min)
    mask.triu_(1)
    return mask


# ---------------------------------------------------------------------------
# Loss A — CLIP cosine similarity (vanilla PEZ, used for warm-start)
# ---------------------------------------------------------------------------


def clip_similarity_loss(
    soft_prompt: torch.Tensor,    # [1, N, D] continuous embedding
    target_image_features: torch.Tensor,    # [1, D] CLIP image embedding
    text_encoder,
    tokenizer,
    text_projection: torch.nn.Linear | None = None,
) -> torch.Tensor:
    """Vanilla PEZ loss: ``-cos_sim(prompt_pooled, image_emb)``.

    Used as the bootstrap loss in PEZ-1's alternating optimization (see
    RESEARCH_PROPOSAL.md §3.1, Round 0).

    Parameters
    ----------
    soft_prompt
        ``[1, N, D]`` — continuous CLIP-input embeddings (the optimization
        variable; gradient flows directly here).
    target_image_features
        ``[1, D]`` — CLIP image encoder's pooled+projected output for
        the source image. Computed once outside the loop.
    text_encoder
        transformers ``CLIPTextModel`` whose token_embedding layer
        produced ``soft_prompt``.
    tokenizer
        Matching ``CLIPTokenizer`` (used for BOS/EOS/PAD ids).
    text_projection
        Optional linear projection to the joint image-text space (the
        ``text_projection`` layer of ``CLIPModel``). If None, compares
        directly in the text encoder's output space, which is fine when
        ``target_image_features`` is also in that space.

    Returns
    -------
    Scalar negative cosine similarity (lower = better match).
    """
    full_embeds, eos_position, position_ids, attention_mask = assemble_77_token_embedding(
        soft_prompt,
        token_embedding=text_encoder.text_model.embeddings.token_embedding,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    _, pooled = encode_through_text_model(
        full_embeds, position_ids, attention_mask, eos_position, text_encoder,
    )

    if text_projection is not None:
        pooled = text_projection(pooled)

    pooled_n = F.normalize(pooled, dim=-1)
    target_n = F.normalize(target_image_features.to(pooled.dtype), dim=-1)
    cos_sim = (pooled_n * target_n).sum(dim=-1).mean()
    return -cos_sim


# ---------------------------------------------------------------------------
# Loss B — CFG-aware SDS with null-text (the project's contribution)
# ---------------------------------------------------------------------------


def sds_cfg_loss(
    soft_prompt: torch.Tensor,              # [1, N, D] continuous embedding
    image_latent: torch.Tensor,             # [1, 4, H, W] VAE-encoded source
    null_text_embedding: torch.Tensor,      # [1, 77, D] frozen null-text at timestep t
    t: torch.Tensor,                        # [1] long, the timestep used for U-Net
    cfg_scale: float,
    unet,                                   # diffusers UNet2DConditionModel
    scheduler,                              # DDIM scheduler with alphas_cumprod
    text_encoder,
    tokenizer,
) -> torch.Tensor:
    """CFG-aware Score Distillation Sampling surrogate of the
    reconstruction loss for the editing pipeline.

    See RESEARCH_PROPOSAL.md §3.1 for the derivation. Two U-Net forward
    passes (cond + uncond); backprop through the cond pass updates the
    soft prompt. The uncond pass is wrapped in ``torch.no_grad()`` since
    null-text is frozen and grad to soft_prompt through it is exactly
    zero — the autograd graph for that pass would just waste memory.

    The caller is responsible for sampling ``t`` and looking up the
    matching ``null_text_embedding``. Decoupling sampling from this
    function makes it possible to use null-text-per-timestep
    consistently: the caller picks t_idx, looks up
    ``null_text_per_timestep[t_idx]``, derives ``t = scheduler.timesteps[t_idx]``,
    and passes both — guaranteeing they're aligned.
    """
    device = soft_prompt.device
    dtype = unet.dtype

    # 1. Encode the soft prompt through CLIP text model to get the
    # 77×768 conditional encoding. This is the prompt embedding that
    # the U-Net's cross-attention will see.
    full_embeds, _, position_ids, attention_mask = assemble_77_token_embedding(
        soft_prompt,
        token_embedding=text_encoder.text_model.embeddings.token_embedding,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Cast embeds to model dtype for U-Net
    full_embeds = full_embeds.to(dtype=dtype)
    eos_position = 1 + soft_prompt.shape[1]
    last_hidden_state, _ = encode_through_text_model(
        full_embeds, position_ids, attention_mask, eos_position, text_encoder,
    )
    # last_hidden_state is [1, 77, D] — the encoder_hidden_states for U-Net.

    # 2. Compute x_t under the caller-provided timestep.
    eps = torch.randn_like(image_latent)
    alpha_cumprod_t = scheduler.alphas_cumprod[t].to(device=device, dtype=dtype)
    sqrt_alpha = alpha_cumprod_t.sqrt()
    sqrt_one_minus_alpha = (1 - alpha_cumprod_t).sqrt()
    x_t = sqrt_alpha * image_latent.to(device=device, dtype=dtype) + sqrt_one_minus_alpha * eps

    # 3. CFG forward pass: two U-Net calls. The unconditional pass is
    # frozen (null-text is detached, x_t/eps don't depend on soft_prompt),
    # so wrap in no_grad to skip building its autograd graph.
    null_text_on_dev = null_text_embedding.to(device=device, dtype=dtype)
    with torch.no_grad():
        eps_uncond = unet(
            x_t, t, encoder_hidden_states=null_text_on_dev,
        ).sample
    eps_cond = unet(
        x_t, t, encoder_hidden_states=last_hidden_state,
    ).sample
    eps_cfg = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    # 4. SDS surrogate loss: MSE against the sampled noise.
    return F.mse_loss(eps_cfg.float(), eps.float())
