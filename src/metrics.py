"""Evaluation metrics: CLIP Dir. Similarity, LPIPS, SSIM, DINO Similarity."""

import torch
import numpy as np
from PIL import Image
import lpips
from skimage.metrics import structural_similarity
import open_clip
from transformers import AutoImageProcessor, AutoModel


def clip_directional_similarity(source_image: Image.Image, target_image: Image.Image, edited_image: Image.Image, device: torch.device = None) -> float:
    """Compute CLIP directional similarity between (edited - source) and (target - source).
    
    Args:
        source_image: Original source image
        target_image: Target image
        edited_image: Edited result image
        device: Device to run on
        
    Returns:
        Cosine similarity score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device).eval()
    
    # Preprocess images
    source_tensor = preprocess(source_image).unsqueeze(0).to(device)
    target_tensor = preprocess(target_image).unsqueeze(0).to(device)
    edited_tensor = preprocess(edited_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        source_emb = model.encode_image(source_tensor)
        target_emb = model.encode_image(target_tensor)
        edited_emb = model.encode_image(edited_tensor)
    
    # Compute directional vectors
    source_to_target = target_emb - source_emb
    source_to_edited = edited_emb - source_emb
    
    # Cosine similarity
    similarity = torch.cosine_similarity(source_to_target, source_to_edited, dim=-1).item()
    
    return similarity


def compute_lpips(image1: Image.Image, image2: Image.Image, device: torch.device = None) -> float:
    """Compute LPIPS distance between two images.
    
    Args:
        image1: First image
        image2: Second image
        device: Device to run on
        
    Returns:
        LPIPS distance (lower is better)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load LPIPS model
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Convert PIL to tensor
    def pil_to_tensor(img):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # Normalize to [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)
    
    tensor1 = pil_to_tensor(image1)
    tensor2 = pil_to_tensor(image2)
    
    with torch.no_grad():
        distance = loss_fn(tensor1, tensor2).item()
    
    return distance


def compute_ssim(image1: Image.Image, image2: Image.Image) -> float:
    """Compute SSIM between two images.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        SSIM score (higher is better)
    """
    # Convert to numpy arrays
    arr1 = np.array(image1).astype(np.float32)
    arr2 = np.array(image2).astype(np.float32)
    
    # Compute SSIM
    ssim_score = structural_similarity(arr1, arr2, channel_axis=2, data_range=255.0)
    
    return ssim_score


def dino_similarity(source_image: Image.Image, target_image: Image.Image, edited_image: Image.Image, device: torch.device = None) -> float:
    """Compute DINO directional similarity between (edited - source) and (target - source).
    
    Args:
        source_image: Original source image
        target_image: Target image
        edited_image: Edited result image
        device: Device to run on
        
    Returns:
        Cosine similarity score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DINOv2 model
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
    
    # Preprocess images
    def process_image(img):
        inputs = processor(images=img, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}
    
    source_inputs = process_image(source_image)
    target_inputs = process_image(target_image)
    edited_inputs = process_image(edited_image)
    
    with torch.no_grad():
        source_outputs = model(**source_inputs)
        target_outputs = model(**target_inputs)
        edited_outputs = model(**edited_inputs)
    
    # Use [CLS] token embedding
    source_emb = source_outputs.last_hidden_state[:, 0, :]
    target_emb = target_outputs.last_hidden_state[:, 0, :]
    edited_emb = edited_outputs.last_hidden_state[:, 0, :]
    
    # Compute directional vectors
    source_to_target = target_emb - source_emb
    source_to_edited = edited_emb - source_emb
    
    # Cosine similarity
    similarity = torch.cosine_similarity(source_to_target, source_to_edited, dim=-1).item()
    
    return similarity


def clip_text_image_similarity(image: Image.Image, text: str, device: torch.device = None) -> float:
    """Compute CLIP similarity between an image and text prompt.
    
    Args:
        image: Input image
        text: Text prompt
        device: Device to run on
        
    Returns:
        Cosine similarity score (higher = more similar)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # Preprocess image and text
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text_tokens = tokenizer([text]).to(device)
    
    with torch.no_grad():
        image_emb = model.encode_image(image_tensor)
        text_emb = model.encode_text(text_tokens)
    
    # Normalize embeddings
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    
    # Cosine similarity
    similarity = torch.matmul(image_emb, text_emb.t()).item()
    
    return similarity
