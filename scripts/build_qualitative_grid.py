"""
Build a qualitative comparison grid from generated images.

Layout: rows = prompts, columns = (model, steps) configurations.
Output: results/figures/qualitative_grid.png

Usage:
  python scripts/build_qualitative_grid.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

IMAGE_BASE = Path("results/images")
PROMPT_FILE = Path("prompts/sample_prompts.txt")
OUT_PATH = Path("results/figures/qualitative_grid.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Columns chosen to tell the speed-quality story:
#   - DDIM at 1 step (failure mode at low steps)
#   - SD1.5+LCM at 1 step (LCM rescues low-step generation)
#   - SDXL-Turbo at 1 step (best 512x512 at 1 step)
#   - SDXL+LCM at 4 steps (high-res sweet spot)
#   - DDIM SDXL at 8 steps (DDIM done right baseline)
COLUMNS = [
    ("ddim_sd15",  1, "DDIM SD1.5\n1 step"),
    ("sd15_lcm",   1, "SD1.5+LCM\n1 step"),
    ("sdxl_turbo", 1, "SDXL-Turbo\n1 step"),
    ("sdxl_lcm",   4, "SDXL+LCM\n4 steps"),
    ("ddim_sdxl",  8, "DDIM SDXL\n8 steps"),
]

PROMPT_LABELS = [
    "futuristic city\nat sunset",
    "golden retriever\nin snow",
    "dragon flying\nover mountains",
    "cabin in\nsnowy forest",
    "robot cooking\nin kitchen",
]


def load_image(model: str, steps: int, idx: int) -> Image.Image:
    path = IMAGE_BASE / model / f"steps_{steps}" / f"{idx:04d}.png"
    return Image.open(path).convert("RGB")


def main():
    n_rows = len(PROMPT_LABELS)
    n_cols = len(COLUMNS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols + 1.2, 2.0 * n_rows),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )

    for row in range(n_rows):
        for col, (model, steps, _) in enumerate(COLUMNS):
            ax = axes[row, col]
            img = load_image(model, steps, row)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row == 0:
                ax.set_title(COLUMNS[col][2], fontsize=9)
            if col == 0:
                ax.set_ylabel(PROMPT_LABELS[row], fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=40)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
