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
    cfg_scale: float = 1.0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run DDIM inversion: encode image latent into noise space.

    Goes forward in diffusion time (t=0 -> t=T), adding noise at each step
    using the model's own noise predictions.

    cfg_scale defaults to 1.0 — Mokady et al. 2022 (null-text inversion)
    and Hertz et al. 2022 (P2P) both require CFG=1.0 for the inversion
    pass. With CFG > 1, each step extrapolates past the conditional
    prediction (eps = (1+s)·eps_cond - s·eps_uncond) instead of
    inverting along the diffusion trajectory, so z_T no longer
    represents the source image — denoising it produces garbage even
    before any P2P injection. Null-text optimization (in the next stage)
    is what compensates for the gap between CFG=1 inversion and CFG=7.5
    denoising.

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
    timestep_idx = int(
        (scheduler.timesteps == timestep).nonzero(as_tuple=True)[0].item()
    )
    # Next timestep in forward diffusion (higher noise).
    #
    # At the highest denoising-step index (idx=0 in scheduler.timesteps
    # order — i.e., the LAST iteration of `reversed(scheduler.timesteps)`),
    # we deliberately set next_timestep = timestep. That makes the
    # update `alpha.sqrt() * pred_x0 + (1-alpha).sqrt() * eps = sample`,
    # i.e., an identity step on the final inversion iteration.
    #
    # This is NOT a missed inversion step. It's the contract with the
    # editing/denoising loop, which iterates scheduler.timesteps in
    # forward order [t_max, ..., t_min] and assumes its starting latent
    # is at noise level alpha[t_max] = alpha[scheduler.timesteps[0]].
    # If we noised past alpha[t_max] (e.g., to alpha[num_train_timesteps - 1]),
    # the very first denoising step would treat that more-noised latent
    # as if it were at alpha[t_max], producing a corrupted reconstruction
    # — visible as washed-out / bad even on the BLIP baseline that
    # bypasses PEZ entirely.
    if timestep_idx == 0:
        next_timestep = int(timestep) if hasattr(timestep, "__int__") else timestep
    else:
        next_timestep = int(scheduler.timesteps[timestep_idx - 1].item())

    # alpha and alpha_prev for the DDIM formula (match sample device + dtype)
    dev, dt = sample.device, sample.dtype
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(dev, dtype=dt)
    alpha_prod_t_next = (
        scheduler.alphas_cumprod[next_timestep].to(dev, dtype=dt)
        if next_timestep >= 0
        else scheduler.final_alpha_cumprod.to(dev, dtype=dt)
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
    # Find previous timestep. nonzero(...) returns a 1-element tensor;
    # collapse to a Python int so prev_timestep stays scalar (a length-1
    # tensor would broadcast in (1 - alpha_prod_prev).sqrt()).
    step_idx = int(
        (scheduler.timesteps == timestep).nonzero(as_tuple=True)[0].item()
    )
    if step_idx + 1 < len(scheduler.timesteps):
        prev_timestep = int(scheduler.timesteps[step_idx + 1].item())
    else:
        prev_timestep = 0

    dev, dt = sample.device, sample.dtype
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(dev, dtype=dt)
    alpha_prod_prev = (
        scheduler.alphas_cumprod[prev_timestep].to(dev, dtype=dt)
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod.to(dev, dtype=dt)
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

        # The conditional noise prediction depends only on `latent` and
        # `text_embeddings` — both constant across the inner loop — so
        # compute it once under no_grad. Doing the conditional inside the
        # inner loop wastes ~half the per-step activation memory because
        # the autograd graph for that branch can never reach uncond_emb.
        latent_in_single = latent.detach()
        with torch.no_grad():
            noise_pred_text = unet(
                latent_in_single,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

        for _ in range(opt_steps):
            # Cast to model dtype for UNet forward pass
            uncond_emb_cast = uncond_emb.to(model_dtype)

            # Only the uncond branch needs a grad-enabled forward.
            noise_pred_uncond = unet(
                latent_in_single,
                t,
                encoder_hidden_states=uncond_emb_cast,
            ).sample

            # Apply CFG
            noise_pred_cfg = noise_pred_uncond + cfg_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Inline DDIM step (differentiable, unlike scheduler.step())
            prev_latent = _ddim_step_inline(
                noise_pred_cfg, t, latent_in_single, scheduler
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
