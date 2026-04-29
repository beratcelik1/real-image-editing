"""Shared utilities: image loading, latent encoding/decoding, model loading."""

from __future__ import annotations

from pathlib import Path


def _patch_hf_chat_template_404() -> None:
    """Workaround for transformers >= ~4.45 mis-handling 404 on chat-template
    lookup for non-LLM tokenizers (CLIPTokenizer in particular).

    transformers calls ``list_repo_tree(repo, "additional_chat_templates")``
    inside ``CLIPTokenizer.from_pretrained``. CLIP doesn't have chat
    templates, so HF returns 404. transformers catches
    ``RepositoryNotFoundError`` internally — but with newer
    huggingface_hub the exception path / class doesn't always match
    the catch, so it escapes to the user as a fatal error even though
    the model is fine.

    Patch ``huggingface_hub.utils._pagination.paginate`` to silently
    return empty for any URL containing ``additional_chat_templates``,
    making the chat-template lookup return [] and letting the load
    proceed. Idempotent; safe to call multiple times.
    """
    try:
        from huggingface_hub.utils import _pagination
    except ImportError:
        return  # very old huggingface_hub; nothing to patch

    if getattr(_pagination, "_chat_template_404_patched", False):
        return  # already patched

    _orig_paginate = _pagination.paginate

    def _patched_paginate(path, params=None, headers=None):
        try:
            yield from _orig_paginate(path, params=params, headers=headers)
        except Exception:
            if isinstance(path, str) and "additional_chat_templates" in path:
                # CLIP and other non-LLM tokenizers don't have chat
                # templates; the 404 is expected. Return empty.
                return
            raise

    _pagination.paginate = _patched_paginate
    _pagination._chat_template_404_patched = True


# Apply the patch BEFORE any transformers import — transformers reads
# huggingface_hub.utils._pagination.paginate at import time in some
# versions. So this needs to run before `from transformers import ...`.
_patch_hf_chat_template_404()


import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_ID_SD21 = "stabilityai/stable-diffusion-2-1-base"
MODEL_ID_SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_SCALE_FACTOR = 0.18215


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """Always float32 — eliminates dtype mismatch bugs. A100 handles it fine."""
    return torch.float32


def load_sd_components(
    device: torch.device | None = None,
    model_id: str | None = None,
) -> tuple[
    UNet2DConditionModel, AutoencoderKL, CLIPTextModel, CLIPTokenizer, DDIMScheduler
]:
    """Load raw SD components (no pipeline wrapper).

    Args:
        device: Target device. Auto-detected if None.
        model_id: HuggingFace model ID. Defaults to SD 2.1, falls back to SD 1.5
                  if auth fails.
    """
    if device is None:
        device = get_device()
    dtype = get_dtype(device)

    if model_id is None:
        model_id = MODEL_ID_SD21
        try:
            tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        except OSError:
            print(f"Cannot access {model_id} (needs HF auth). Falling back to SD 1.5.")
            model_id = MODEL_ID_SD15
            tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    else:
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    print(f"Using model: {model_id}")

    text_encoder = (
        CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    vae = (
        AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
        .to(device)
        .eval()
    )
    unet = (
        UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # The optimization variable is the soft prompt; SD's component
    # weights are never updated. Without requires_grad_=False the
    # autograd graph for every PEZ-1/PEZ-2 SDS step keeps activations
    # alive for ~all of text_encoder/vae/unet — wasted memory.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    return unet, vae, text_encoder, tokenizer, scheduler


def load_image(path: str | Path, size: int = 512) -> Image.Image:
    """Load and resize image to square."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return img


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL Image -> (1, 3, H, W) tensor normalized to [-1, 1]."""
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def encode_image(
    image: Image.Image, vae: AutoencoderKL, device: torch.device
) -> torch.Tensor:
    """Encode PIL image to VAE latent space."""
    pixel_values = image_to_tensor(image, device).to(vae.dtype)
    with torch.no_grad():
        latent_dist = vae.encode(pixel_values).latent_dist
        latent = latent_dist.mean * VAE_SCALE_FACTOR
    return latent


def decode_latent(latent: torch.Tensor, vae: AutoencoderKL) -> Image.Image:
    """Decode VAE latent to PIL image."""
    with torch.no_grad():
        image = vae.decode(latent / VAE_SCALE_FACTOR).sample
    # (1, 3, H, W) -> PIL
    image = image.clamp(-1, 1)
    image = (image[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) / 2.0
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def get_text_embeddings(
    prompt: str,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    """Encode text prompt to CLIP hidden states."""
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids.to(device))[0]
    return embeddings


def get_uncond_embeddings(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    """Get unconditional (empty prompt) embeddings."""
    return get_text_embeddings("", tokenizer, text_encoder, device)
