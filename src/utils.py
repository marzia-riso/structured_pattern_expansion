import clip
import torch

import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_PATH = r".\resources\mask.png"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
IMAGE_SIZE = 512

def mask_to_box(mask):
    """Extracts the bounding box from the mask"""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.array([0, 0, mask.shape[1], mask.shape[0]]) # Fallback full mask
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])

def extract_guidance_image(image, mask_image):
    """Extract the guidance image from the input image"""
    mask = np.array(mask_image.convert("L"), dtype=np.uint8) == 0
    box = mask_to_box(mask)

    patch = image.crop(box)

    offset_width = mask_image.width//2 - patch.width//2
    offset_height = mask_image.height//2 - patch.height//2

    ver_reps = int(np.ceil(mask_image.height / patch.height))
    hor_reps = int(np.ceil(mask_image.width / patch.width))

    guidance_image = np.tile(patch, (ver_reps, hor_reps, 1))
    guidance_image = np.roll(guidance_image, shift=(offset_width, offset_height), axis=(0, 1))
    guidance_image = Image.fromarray(guidance_image).convert("RGB").crop((0, 0, mask_image.width, mask_image.height))   
    return guidance_image

def compute_clip_embeddings(images):
    """Computes the CLIP embeddings for the images"""
    model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    images = torch.stack([preprocess(img) for img in images], dim=0).to(DEVICE)

    with torch.no_grad():
        embs = model.encode_image(images)
    return embs

def compute_cross_similarity(guidance, generated_images):
    """Computes the cross similarity between the guidance and generated images"""
    guidance_embs = compute_clip_embeddings(guidance)
    guidance_embs = guidance_embs / guidance_embs.norm(dim=-1, keepdim=True)

    generated_embs = compute_clip_embeddings(generated_images)
    generated_embs = generated_embs / generated_embs.norm(dim=-1, keepdim=True)

    cross_similarity = guidance_embs @ generated_embs.T
    return cross_similarity

def save_images(image, image_name, output_dir):
    """Saves the images to the output directory"""
    image.save(output_dir / (image_name + ".png"))