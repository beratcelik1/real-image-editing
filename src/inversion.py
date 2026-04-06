"""DDIM inversion and Null-text optimization for real image editing."""

from __future__ import annotations

import torch
from tqdm import tqdm
from diffusers import DDIMScheduler, UNet2DConditionModel


def ddim_inversion(
    latent: torch.Tensor,
    text_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    num_steps: int = 50,
    cfg_scale: float = 7.5,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run DDIM inversion: encode image latent into noise space.

    Goes forward in diffusion time (t=0 -> t=T), adding noise at each step
    using the model's own noise predictions.

    Returns:
        latent_T: The fully inverted noisy latent at timestep T.
        latents_trajectory: List of latents at each timestep [z_0, z_1, ..., z_T].
    """
    scheduler.set_timesteps(num_steps)
    # Reverse timesteps: we go from t=0 toward t=T
    timesteps = reversed(scheduler.timesteps)

    all_latents = [latent.clone()]

    latent = latent.clone()
    for t in tqdm(timesteps, desc="DDIM Inversion"):
        # Duplicate latent for CFG (uncond + cond)
        latent_model_input = torch.cat([latent] * 2)
        embed_input = torch.cat([uncond_embeddings, text_embeddings])

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embed_input,
            ).sample

        # Apply classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # DDIM inversion step: go forward in noise (reverse of denoising)
        latent = _ddim_inversion_step(scheduler, noise_pred, t, latent)
        all_latents.append(latent.clone())

    return latent, all_latents


def _ddim_inversion_step(
    scheduler: DDIMScheduler,
    noise_pred: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    """Single DDIM inversion step: x_t -> x_{t+1}.

    Reverses the DDIM denoising formula to add noise.
    """
    # Get the next timestep (one step further into noise)
    prev_timestep = timestep
    timestep_idx = (scheduler.timesteps == timestep).nonzero(as_tuple=True)[0]
    # Next timestep in forward diffusion (higher noise)
    if timestep_idx == 0:
        next_timestep = timestep  # Already at highest noise
    else:
        next_timestep = scheduler.timesteps[timestep_idx - 1]

    # alpha and alpha_prev for the DDIM formula (move to sample device)
    dev = sample.device
    alpha_prod_t = scheduler.alphas_cumprod[prev_timestep].to(dev)
    alpha_prod_t_next = (
        scheduler.alphas_cumprod[next_timestep].to(dev)
        if next_timestep >= 0
        else scheduler.final_alpha_cumprod.to(dev)
    )

    # Predict x_0 from current sample
    pred_x0 = (sample - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()

    # Compute x_{t+1} (more noisy) using DDIM formula in reverse
    direction = (1 - alpha_prod_t_next).sqrt() * noise_pred
    noisy_sample = alpha_prod_t_next.sqrt() * pred_x0 + direction

    return noisy_sample


def _ddim_step_inline(
    noise_pred: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    scheduler: DDIMScheduler,
) -> torch.Tensor:
    """Differentiable DDIM denoising step (inlined, no scheduler.step()).

    Needed because scheduler.step() may break the gradient graph.
    """
    # Find previous timestep
    step_idx = (scheduler.timesteps == timestep).nonzero(as_tuple=True)[0]
    if step_idx + 1 < len(scheduler.timesteps):
        prev_timestep = scheduler.timesteps[step_idx + 1]
    else:
        prev_timestep = 0

    dev = sample.device
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(dev)
    alpha_prod_prev = (
        scheduler.alphas_cumprod[prev_timestep].to(dev)
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod.to(dev)
    )

    # Predict x_0
    pred_x0 = (sample - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()

    # Compute x_{t-1}
    direction = (1 - alpha_prod_prev).sqrt() * noise_pred
    prev_sample = alpha_prod_prev.sqrt() * pred_x0 + direction

    return prev_sample


def null_text_optimization(
    latents_trajectory: list[torch.Tensor],
    text_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    num_steps: int = 50,
    cfg_scale: float = 7.5,
    opt_steps: int = 10,
    lr: float = 1e-2,
) -> list[torch.Tensor]:
    """Optimize unconditional embeddings for faithful reconstruction.

    For each timestep, optimizes the null-text embedding so that DDIM
    denoising with CFG reconstructs the original image perfectly.

    Args:
        latents_trajectory: [z_0, z_1, ..., z_T] from DDIM inversion.
        text_embeddings: Conditional text embeddings for the source prompt.
        uncond_embeddings: Initial unconditional embeddings (CLIP encoding of "").
        unet: The UNet model (frozen).
        scheduler: DDIM scheduler.
        num_steps: Number of diffusion steps.
        cfg_scale: Classifier-free guidance scale.
        opt_steps: Gradient descent steps per timestep.
        lr: Learning rate for null-text optimization.

    Returns:
        null_embeddings: List of optimized unconditional embeddings, one per timestep.
    """
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    # Freeze UNet to avoid OOM from retaining its computation graph
    unet.requires_grad_(False)

    uncond_embedding_init = uncond_embeddings.clone()
    null_embeddings = []

    # Start from the noisiest latent and denoise step by step
    latent = latents_trajectory[-1].clone()

    # Optimize in float32 for numerical stability (Adam + float16 = bad)
    opt_dtype = torch.float32
    model_dtype = text_embeddings.dtype

    for i, t in enumerate(tqdm(timesteps, desc="Null-text Optimization")):
        # Target: the pivot latent at this timestep from inversion
        # latents_trajectory goes [z_0, ..., z_T], timesteps go [T, ..., 0]
        target_latent = latents_trajectory[len(timesteps) - 1 - i].clone().detach()

        # Initialize learnable null embedding for this timestep (float32 for optimization)
        uncond_emb = (
            uncond_embedding_init.clone().detach().to(opt_dtype).requires_grad_(True)
        )
        optimizer = torch.optim.Adam([uncond_emb], lr=lr)

        for _ in range(opt_steps):
            # Cast to model dtype for UNet forward pass
            uncond_emb_cast = uncond_emb.to(model_dtype)

            # Predict noise with current null embedding
            latent_input = torch.cat([latent.detach()] * 2)
            embed_input = torch.cat([uncond_emb_cast, text_embeddings])

            noise_pred = unet(
                latent_input,
                t,
                encoder_hidden_states=embed_input,
            ).sample

            # Apply CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + cfg_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Inline DDIM step (differentiable, unlike scheduler.step())
            prev_latent = _ddim_step_inline(
                noise_pred_cfg, t, latent.detach(), scheduler
            )

            # Loss: predicted latent should match the pivot from inversion
            loss = torch.nn.functional.mse_loss(
                prev_latent.float(), target_latent.float()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Store optimized embedding (in model dtype) and advance latent
        null_embeddings.append(uncond_emb.detach().to(model_dtype))
        uncond_embedding_init = uncond_emb.detach().to(model_dtype)

        # Take a DDIM step with the optimized null embedding (no grad)
        with torch.no_grad():
            latent_input = torch.cat([latent] * 2)
            embed_input = torch.cat(
                [uncond_emb.detach().to(model_dtype), text_embeddings]
            )
            noise_pred = unet(latent_input, t, encoder_hidden_states=embed_input).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + cfg_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latent = scheduler.step(noise_pred_cfg, t, latent).prev_sample

    return null_embeddings


def reconstruct(
    latent_T: torch.Tensor,
    null_embeddings: list[torch.Tensor],
    text_embeddings: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    num_steps: int = 50,
    cfg_scale: float = 7.5,
) -> torch.Tensor:
    """DDIM denoising with optimized null embeddings for reconstruction.

    Returns the reconstructed latent (should closely match the original).
    """
    scheduler.set_timesteps(num_steps)
    latent = latent_T.clone()

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Reconstruction")):
        uncond_emb = null_embeddings[i]
        latent_input = torch.cat([latent] * 2)
        embed_input = torch.cat([uncond_emb, text_embeddings])

        with torch.no_grad():
            noise_pred = unet(latent_input, t, encoder_hidden_states=embed_input).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_cfg = noise_pred_uncond + cfg_scale * (
            noise_pred_text - noise_pred_uncond
        )

        latent = scheduler.step(noise_pred_cfg, t, latent).prev_sample

    return latent
