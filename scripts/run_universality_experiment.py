"""
Verify the LCM-LoRA universality claim: a single LCM-LoRA adapter combined
with an arbitrary style LoRA via tau' = lambda_1 * tau_style + lambda_2 * tau_LCM
produces stylized images at 4 steps without retraining.

Generates three image sets on a fixed set of prompts:
  - SDXL + style LoRA  @ 30 steps DDIM   (style reference)
  - SDXL + style LoRA  @  4 steps DDIM   (style without acceleration)
  - SDXL + style LoRA + LCM-LoRA @ 4 steps LCM (the claim)

And assembles results/figures/universality_grid.png.

Style LoRA: nerijs/pixel-art-xl (pixel-art style on SDXL).

Usage:
  python scripts/run_universality_experiment.py
"""

from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt

from model_loader import load_pipeline


PROMPTS = [
    "a futuristic city skyline at sunset",
    "a golden retriever playing in snow",
    "a dragon flying over mountains",
    "a cozy wooden cabin in a snowy forest",
    "a robot cooking dinner in a kitchen",
]

OUT_BASE = Path("results/images")
OUT_FIG = Path("results/figures/universality_grid.png")
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    # (model_name, steps, guidance, label_for_grid, dir_name)
    ("sdxl_styled",     30, 7.5, "SDXL+Style\n30 steps DDIM",   "sdxl_styled_30"),
    ("sdxl_styled",      4, 7.5, "SDXL+Style\n4 steps DDIM",    "sdxl_styled_4"),
    ("sdxl_styled_lcm",  4, 1.0, "SDXL+Style+LCM\n4 steps LCM", "sdxl_styled_lcm_4"),
]

WIDTH = HEIGHT = 1024
SEED = 42


def generate_for_config(model_name: str, steps: int, guidance: float, dir_name: str):
    save_dir = OUT_BASE / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(model_name)
    device = pipe.device

    for idx, prompt in enumerate(PROMPTS):
        out_path = save_dir / f"{idx:04d}.png"
        if out_path.exists():
            print(f"  skip (exists): {out_path}")
            continue

        styled_prompt = f"pixel art, {prompt}"
        generator = torch.Generator(device=device).manual_seed(SEED + idx)

        image = pipe(
            prompt=styled_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            width=WIDTH,
            height=HEIGHT,
        ).images[0]
        image.save(out_path)
        print(f"  saved {out_path}")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_grid():
    n_rows = len(PROMPTS)
    n_cols = len(CONFIGS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols + 1.4, 2.0 * n_rows),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )

    for row in range(n_rows):
        for col, (_, _, _, label, dir_name) in enumerate(CONFIGS):
            ax = axes[row, col]
            img_path = OUT_BASE / dir_name / f"{row:04d}.png"
            ax.imshow(Image.open(img_path).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row == 0:
                ax.set_title(label, fontsize=9)
            if col == 0:
                short = PROMPTS[row].split(",")[0][:24]
                ax.set_ylabel(short, fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=40)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_FIG.resolve()}")


def main():
    for model_name, steps, guidance, _, dir_name in CONFIGS:
        print(f"\n=== {dir_name} ({model_name}, {steps} steps, guidance={guidance}) ===")
        generate_for_config(model_name, steps, guidance, dir_name)

    print("\nBuilding universality grid ...")
    build_grid()


if __name__ == "__main__":
    main()
