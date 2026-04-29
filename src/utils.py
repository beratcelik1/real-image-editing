"""Shared utilities: image loading, latent encoding/decoding, model loading."""

from __future__ import annotations

from pathlib import Path


def _patch_hf_chat_template_404() -> None:
    """Workaround for transformers >= ~4.45 mis-handling 404 on chat-template
    lookup for non-LLM tokenizers (CLIPTokenizer in particular).

    transformers calls ``list_repo_tree(repo, "additional_chat_templates")``
    inside ``CLIPTokenizer.from_pretrained``. CLIP doesn't have chat
    templates, so HF returns 404. The exception escapes to the user as
    a fatal error even though the model is fine.

    Why this is hard to patch correctly: ``huggingface_hub.hf_api`` does
    ``from .utils._pagination import paginate`` at module-import time,
    binding ``paginate`` as a *local* name in ``hf_api``. Patching
    ``_pagination.paginate`` after that does NOT update ``hf_api``'s
    local reference. Same for ``list_repo_tree`` if a caller imports it
    as a free function.

    The most reliable fix is to short-circuit at the TOP — replace
    ``transformers.utils.hub.list_repo_templates`` to always return
    empty. CLIP tokenizers have no chat templates, so empty is the
    correct return value. We also patch lower layers as belt-and-
    suspenders.

    Idempotent; safe to call multiple times.
    """
    # Layer 0 (most reliable): replace `list_repo_templates` with a
    # function that returns []. CLIP tokenizers genuinely have no chat
    # templates, so [] is the correct return value.
    #
    # Critical detail: must patch in BOTH locations because
    # tokenization_utils_base.py imports it via `from .utils.hub import
    # list_repo_templates` at *its* module-import time, binding a local
    # name. Patching only utils.hub.list_repo_templates leaves
    # tokenization_utils_base.list_repo_templates as the original
    # function — which is what from_pretrained actually calls.
    def _no_chat_templates(*args, **kwargs):
        return []

    for module_path in (
        "transformers.utils.hub",
        "transformers.tokenization_utils_base",
    ):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            if not getattr(mod, "_chat_template_404_patched", False):
                if hasattr(mod, "list_repo_templates"):
                    mod.list_repo_templates = _no_chat_templates
                mod._chat_template_404_patched = True
        except ImportError:
            pass  # transformers not installed; nothing to patch

    # Layer 1: patch huggingface_hub.hf_api.HfApi.list_repo_tree —
    # in case anything else calls it for additional_chat_templates.
    try:
        from huggingface_hub import hf_api
    except ImportError:
        return

    if not getattr(hf_api, "_chat_template_404_patched", False):
        _orig_list_repo_tree = hf_api.HfApi.list_repo_tree

        def _patched_list_repo_tree(self, repo_id, path_in_repo=None, **kwargs):
            try:
                yield from _orig_list_repo_tree(
                    self, repo_id, path_in_repo=path_in_repo, **kwargs,
                )
            except Exception:
                if (
                    isinstance(path_in_repo, str)
                    and "additional_chat_templates" in path_in_repo
                ):
                    return  # silently empty
                raise

        hf_api.HfApi.list_repo_tree = _patched_list_repo_tree
        hf_api._chat_template_404_patched = True

    # Layer 2: patch the module-level paginate in BOTH locations — the
    # canonical _pagination.paginate AND the local copy in hf_api (which
    # was bound at hf_api's import time via `from .utils._pagination
    # import paginate`).
    try:
        from huggingface_hub.utils import _pagination
    except ImportError:
        return

    if not getattr(_pagination, "_chat_template_404_patched", False):
        _orig_paginate = _pagination.paginate

        def _patched_paginate(path, params=None, headers=None):
            try:
                yield from _orig_paginate(path, params=params, headers=headers)
            except Exception:
                if isinstance(path, str) and "additional_chat_templates" in path:
                    return  # silently empty
                raise

        _pagination.paginate = _patched_paginate
        # Also patch hf_api's local reference if it has one — this is the
        # critical step the prior commit missed.
        if hasattr(hf_api, "paginate"):
            hf_api.paginate = _patched_paginate
        _pagination._chat_template_404_patched = True


# Apply the patch BEFORE any transformers import. The Layer-0 patch
# above relies on `transformers` being importable, so this must run
# AFTER `import transformers` is possible but BEFORE any code triggers
# the chat-template lookup. By calling here at module-import time of
# src/utils.py, we ensure the patch is in place before anything in
# this module (or downstream) loads CLIPTokenizer.
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
