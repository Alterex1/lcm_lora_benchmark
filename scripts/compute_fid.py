import argparse
import csv
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import inception_v3


def get_inception_model(device: str):
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove final classification layer
    model = model.to(device)
    model.eval()
    return model


def preprocess_images(image_dir: str, image_size: int = 299) -> List[Image.Image]:
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(".png")]
    )

    images = []
    for f in image_files:
        img = Image.open(os.path.join(image_dir, f)).convert("RGB")
        images.append(img)

    return images


def extract_features(
    images: List[Image.Image],
    model: torch.nn.Module,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        batch_tensors = torch.stack([transform(img) for img in batch_images]).to(device)

        with torch.no_grad():
            features = model(batch_tensors)

        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def compute_fid(features_1: np.ndarray, features_2: np.ndarray) -> float:
    mu1, sigma1 = features_1.mean(axis=0), np.cov(features_1, rowvar=False)
    mu2, sigma2 = features_2.mean(axis=0), np.cov(features_2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


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
        description="Compute FID scores between generated images and a reference set."
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="results/images",
        help="Base directory containing model/steps subdirectories of images.",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        required=True,
        help="Directory containing reference (real) images for FID comparison.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/metrics/fid_scores.csv",
        help="CSV file to append FID results to.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for feature extraction.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.reference_dir):
        print(f"Error: reference directory {args.reference_dir} does not exist.")
        return

    if not os.path.exists(args.image_base_dir):
        print(f"Error: {args.image_base_dir} does not exist. Run generate_images.py first.")
        return

    print("Loading Inception v3 model...")
    model = get_inception_model(device)

    print("Extracting features from reference images...")
    ref_images = preprocess_images(args.reference_dir)
    if len(ref_images) < 2:
        print("Error: need at least 2 reference images for FID computation.")
        return
    ref_features = extract_features(ref_images, model, device, args.batch_size)
    print(f"  Extracted features from {len(ref_images)} reference images.")

    for model_name in sorted(os.listdir(args.image_base_dir)):
        model_dir = os.path.join(args.image_base_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        for steps_dir_name in sorted(os.listdir(model_dir)):
            steps_dir = os.path.join(model_dir, steps_dir_name)
            if not os.path.isdir(steps_dir):
                continue

            try:
                steps = int(steps_dir_name.split("_")[1])
            except (IndexError, ValueError):
                continue

            print(f"Computing FID for {model_name} at {steps} steps...")
            gen_images = preprocess_images(steps_dir)

            if len(gen_images) < 2:
                print(f"  Skipping: need at least 2 generated images, found {len(gen_images)}.")
                continue

            gen_features = extract_features(gen_images, model, device, args.batch_size)
            fid_score = compute_fid(ref_features, gen_features)

            row = {
                "model_name": model_name,
                "steps": steps,
                "num_generated": len(gen_images),
                "num_reference": len(ref_images),
                "fid_score": round(fid_score, 4),
            }

            append_csv_row(args.output_csv, row)
            print(f"  FID: {fid_score:.4f} ({len(gen_images)} generated vs {len(ref_images)} reference)")

    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
