import argparse
import csv
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_prompts(prompt_file: str, max_prompts: int | None = None) -> List[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    return prompts


def compute_clip_scores(
    image_dir: str,
    prompts: List[str],
    model_name: str = "openai/clip-vit-large-patch14",
) -> List[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(".png")],
    )

    if len(image_files) != len(prompts):
        print(
            f"Warning: found {len(image_files)} images but {len(prompts)} prompts. "
            f"Using min of both."
        )

    count = min(len(image_files), len(prompts))
    scores = []

    with torch.no_grad():
        for i in range(count):
            image = Image.open(os.path.join(image_dir, image_files[i])).convert("RGB")
            prompt = prompts[i]

            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(device)

            outputs = model(**inputs)

            # Cosine similarity between image and text embeddings, scaled to 0-100
            score = outputs.logits_per_image.item()
            scores.append(score)

    return scores


def append_csv_row(csv_path: str, row: dict) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CLIP scores for generated images."
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="results/images",
        help="Base directory containing model/steps subdirectories of images.",
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
        "--output_csv",
        type=str,
        default="results/metrics/clip_scores.csv",
        help="CSV file to append CLIP score results to.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    prompts = load_prompts(args.prompt_file, args.max_prompts)
    print(f"Loaded {len(prompts)} prompts")

    if not os.path.exists(args.image_base_dir):
        print(f"Error: {args.image_base_dir} does not exist. Run generate_images.py first.")
        return

    # Walk through model/steps_N directories
    for model_name in sorted(os.listdir(args.image_base_dir)):
        model_dir = os.path.join(args.image_base_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        for steps_dir_name in sorted(os.listdir(model_dir)):
            steps_dir = os.path.join(model_dir, steps_dir_name)
            if not os.path.isdir(steps_dir):
                continue

            # Extract step count from directory name (e.g., "steps_4" -> 4)
            try:
                steps = int(steps_dir_name.split("_")[1])
            except (IndexError, ValueError):
                continue

            print(f"Computing CLIP scores for {model_name} at {steps} steps...")
            scores = compute_clip_scores(steps_dir, prompts)

            if not scores:
                print(f"  No scores computed, skipping.")
                continue

            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            row = {
                "model_name": model_name,
                "steps": steps,
                "num_images": len(scores),
                "avg_clip_score": round(avg_score, 4),
                "min_clip_score": round(min_score, 4),
                "max_clip_score": round(max_score, 4),
            }

            append_csv_row(args.output_csv, row)
            print(f"  Avg CLIP score: {avg_score:.4f} ({len(scores)} images)")

    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
