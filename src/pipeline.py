"""Main editing pipeline: inversion -> attention extraction -> editing."""

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.pipeline <image_path> <prompt>")
        print('Example: python -m src.pipeline data/cat.jpg "a photo of a cat"')
        sys.exit(1)

    invert_and_reconstruct(sys.argv[1], sys.argv[2])
