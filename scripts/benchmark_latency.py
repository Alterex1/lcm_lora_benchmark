import argparse
import csv
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import torch

# Assumes your team creates this file.
# It should return a ready-to-run Diffusers pipeline on the correct device.
from model_loader import load_pipeline


def load_prompts(prompt_file: str, max_prompts: int | None = None) -> List[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")

    return prompts


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    ensure_parent_dir(csv_path)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(row)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def reset_peak_memory_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: str) -> float:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def infer_resolution(model_name: str) -> int:
    if model_name == "sdxl_turbo":
        return 512
    if "sdxl" in model_name:
        return 1024
    return 512


def generate_one(
    pipe,
    prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        width=width,
        height=height,
    )
    return result.images[0]


def benchmark(
    pipe,
    prompts: List[str],
    model_name: str,
    method_name: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    warmup_runs: int,
    output_csv: str,
    base_seed: int,
    save_images: bool = False,
    save_dir: str | None = None,
) -> Dict[str, Any]:
    device = str(pipe.device)

    # Warmup
    for i in range(warmup_runs):
        _ = generate_one(
            pipe=pipe,
            prompt=prompts[i % len(prompts)],
            steps=steps,
            guidance_scale=guidance_scale,
            seed=base_seed + i,
            width=width,
            height=height,
        )
        synchronize_if_needed(device)

    reset_peak_memory_if_needed(device)

    per_image_latencies = []
    total_start = time.perf_counter()

    if save_images and save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        synchronize_if_needed(device)
        start = time.perf_counter()

        image = generate_one(
            pipe=pipe,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=base_seed + idx,
            width=width,
            height=height,
        )

        synchronize_if_needed(device)
        end = time.perf_counter()

        latency = end - start
        per_image_latencies.append(latency)

        if save_images and save_dir is not None:
            image_path = os.path.join(save_dir, f"{idx:04d}.png")
            image.save(image_path)

    synchronize_if_needed(device)
    total_end = time.perf_counter()

    total_time = total_end - total_start
    avg_latency = sum(per_image_latencies) / len(per_image_latencies)
    throughput = len(prompts) / total_time if total_time > 0 else 0.0
    peak_mem_mb = get_peak_memory_mb(device)

    row = {
        "model_name": model_name,
        "method_name": method_name,
        "device": device,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "num_prompts": len(prompts),
        "warmup_runs": warmup_runs,
        "total_time_sec": round(total_time, 6),
        "avg_latency_sec": round(avg_latency, 6),
        "min_latency_sec": round(min(per_image_latencies), 6),
        "max_latency_sec": round(max(per_image_latencies), 6),
        "throughput_img_per_sec": round(throughput, 6),
        "peak_memory_mb": round(peak_mem_mb, 2),
    }

    append_csv_row(output_csv, row)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark image generation latency for diffusion pipelines.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["sd15_lcm", "sdxl_lcm", "ddim_sd15", "ddim_sdxl", "sdxl_turbo"],
        help="Which pipeline to benchmark.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of inference steps (e.g. 1, 2, 4, 8).",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to text file with one prompt per line.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optional limit on number of prompts.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Override guidance scale. If omitted, a model-specific default is used.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width. If omitted, a model-specific default is used.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height. If omitted, a model-specific default is used.",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=1,
        help="Number of warmup generations before timing.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/metrics/latency_results.csv",
        help="CSV file to append benchmark results to.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Whether to save generated images during benchmarking.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save generated images if --save_images is set.",
    )

    return parser.parse_args()


def get_default_guidance_scale(model_name: str) -> float:
    if model_name in {"sd15_lcm", "sdxl_lcm"}:
        return 1.0
    if model_name == "sdxl_turbo":
        return 0.0
    if model_name in {"ddim_sd15", "ddim_sdxl"}:
        return 7.5
    return 7.5


def main():
    args = parse_args()

    prompts = load_prompts(args.prompt_file, args.max_prompts)

    width = args.width if args.width is not None else infer_resolution(args.model)
    height = args.height if args.height is not None else infer_resolution(args.model)
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else get_default_guidance_scale(args.model)
    )

    print(f"Loading pipeline for model: {args.model}")
    pipe = load_pipeline(args.model)

    print(f"Loaded {len(prompts)} prompts")
    print(f"Running benchmark: model={args.model}, steps={args.steps}, size={width}x{height}")

    row = benchmark(
        pipe=pipe,
        prompts=prompts,
        model_name=args.model,
        method_name=args.model,
        steps=args.steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        warmup_runs=args.warmup_runs,
        output_csv=args.output_csv,
        base_seed=args.seed,
        save_images=args.save_images,
        save_dir=args.save_dir,
    )

    print("\nBenchmark complete:")
    for k, v in row.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()