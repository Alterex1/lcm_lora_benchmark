import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch

from model_loader import load_pipeline


MODELS = ["sd15_lcm", "sdxl_lcm", "ddim_sd15", "ddim_sdxl", "sdxl_turbo"]
STEPS = [1, 2, 4, 8]


def load_prompts(prompt_file: str, max_prompts: int | None = None) -> List[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")

    return prompts


def get_default_guidance_scale(model_name: str) -> float:
    if model_name in {"sd15_lcm", "sdxl_lcm"}:
        return 1.0
    if model_name == "sdxl_turbo":
        return 0.0
    return 7.5


def infer_resolution(model_name: str) -> int:
    if model_name == "sdxl_turbo":
        return 512
    if "sdxl" in model_name:
        return 1024
    return 512


def generate_images(
    pipe,
    prompts: List[str],
    model_name: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    output_dir: str,
    seed: int = 42,
):
    save_dir = os.path.join(output_dir, model_name, f"steps_{steps}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    device = pipe.device

    for idx, prompt in enumerate(prompts):
        generator = torch.Generator(device=device).manual_seed(seed + idx)

        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
        ).images[0]

        image_path = os.path.join(save_dir, f"{idx:04d}.png")
        image.save(image_path)
        print(f"  Saved {image_path}")

    return save_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images for all model/step combinations."
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompts/sample_prompts.txt",
        help="Path to text file with one prompt per line.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optional limit on number of prompts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/images",
        help="Base directory to save generated images.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Which models to generate images for.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=STEPS,
        help="Inference step counts to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    prompts = load_prompts(args.prompt_file, args.max_prompts)
    print(f"Loaded {len(prompts)} prompts")

    for model_name in args.models:
        print(f"\nLoading {model_name}...")
        pipe = load_pipeline(model_name)

        width = infer_resolution(model_name)
        height = width
        guidance_scale = get_default_guidance_scale(model_name)

        valid_steps = [s for s in args.steps if s <= 4] if model_name == "sdxl_turbo" else args.steps

        for steps in valid_steps:
            print(f"  Generating with {steps} steps (guidance={guidance_scale}, {width}x{height})...")
            generate_images(
                pipe=pipe,
                prompts=prompts,
                model_name=model_name,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                output_dir=args.output_dir,
                seed=args.seed,
            )

        # Free GPU memory before loading next model
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nImage generation complete.")


if __name__ == "__main__":
    main()
