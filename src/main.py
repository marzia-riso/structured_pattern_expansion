import argparse
from pathlib import Path
from datetime import datetime

from PIL import Image
from utils import *
from models import expansion_pipeline

import torch

def arg_parser():
    """Parse the CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", "-i", type=Path, required=True, help="Path of the image to be expanded or a directory containing images")
    parser.add_argument("--output_dir", "-o", type=Path, default=Path(f"./results/test-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"), help="Path of the output directory")
    parser.add_argument("--num_images", "-n", type=int, default=4, help="Number of images to be generated")
    parser.add_argument("--expansion_factor", "-f", type=int, nargs="+", default=[1, 1], help="Expansion factor. It can be one or two ints",)
    return parser.parse_args()

def args_validate(args):
    """Validate the arguments"""
    if not args.input_data.exists():
        raise FileNotFoundError(f"Invalid input data path: {args.input_data}")
    if args.num_images <= 0:
        raise ValueError("-n expects a positive integer")
    if len(args.expansion_factor) != 1 and len(args.expansion_factor) != 2:
        raise ValueError("-f expects 1 or 2 integers: [expansion_factor] or [expansion_factor expansion_factor]")

    if args.input_data.is_dir():
        # Collect the images from the directory
        args.input_data = [Path(p) for p in args.input_data.iterdir() if p.suffix.lower() in IMAGE_EXTS]        
        if len(args.input_data) == 0:
                raise FileNotFoundError(f"No images found in the directory: {args.input_data}")
    else:
        # Validate the image extension
        if args.input_data.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension for --input_data: {args.input_data.suffix}")
        
        args.input_data = [args.input_data]
        
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

def log_args(args):
    """Logging the CLI arguments"""
    print(f"[LOG] Generation started at {datetime.now().strftime('%d-%m-%Y-%H-%M')}")
    print(f"----- Output directory: {args.output_dir}")
    print(f"----- Number of images: {args.num_images}")
    print(f"----- Expansion factor: {args.expansion_factor}")

def main_single_expansion(pipeline, init_image, output_dir, num_images, expansion_factor):
    """Expansion function. 
        It takes the <input image> and produces <number of images> expanded images with an <expansion factor>.
        The results are saved in the <output directory>.
    """
    logged_images = {}

    # Load the input images 
    logged_images['init_image'] = Image.open(init_image).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    logged_images['mask_image'] = Image.open(MASK_PATH).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    logged_images['guidance_image'] = extract_guidance_image(logged_images['init_image'], mask_image=logged_images['mask_image'])

    # Define the prompt and negative prompt
    prompt = "Regular hand-drawn repeating pattern with flat colors." 
    negative_prompt = "Blurry. Undetailed. Irregular. Low Quality. Noisy. Unstructured."

    # Generate the images through the expansion pipeline
    output = pipeline(
        prompt = prompt, # Text prompt reinforcing expansion
        negative_prompt = negative_prompt, # Text negative prompt discouraging low quality images
        image = logged_images['init_image'], # Image to be expanded
        mask_image = logged_images['mask_image'], # Mask image
        ip_adapter_image = logged_images['guidance_image'], # Guidance image
        num_inference_steps = 50, # Number of inference steps to denoise the image
        guidance_scale = 6, # Guidance scale
        eta = 1.0, # Eta
        num_images_per_prompt = num_images, # Number of images to generate
        cross_attention_kwargs = {"scale": 0.75}, # Cross attention kwargs to control the strength of the guidance
        replication_iter = 30, # Iteration threshold for image expansion
        expansion_factor = expansion_factor # Expansion factor
    ).images 

    # Compute the cross similarity between the guidance and generated images
    similarities = compute_cross_similarity([logged_images['guidance_image']] * num_images, output)[0]
    _, indices = torch.sort(similarities, descending=True)

    for i, index in enumerate(indices):
        logged_images[f"result_{i}"] = output[index]

    # Save the images
    for image_name, image in logged_images.items():
        save_images(image, image_name, output_dir)

def main_expansion(args):
    log_args(args)

    with torch.no_grad():
        pipeline = expansion_pipeline()

        for image in args.input_data:
            print(f"----- Processing image: {image}")
            image_output_dir = args.output_dir / image.stem
            if not image_output_dir.exists():
                image_output_dir.mkdir(parents=True, exist_ok=True)

            main_single_expansion(pipeline, image, image_output_dir, args.num_images, args.expansion_factor)

if __name__ == "__main__":
    args = arg_parser()
    args_validate(args)

    main_expansion(args)
