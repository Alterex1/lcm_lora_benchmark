"""
Download COCO 2017 captions and images for use as the benchmark prompt set
and the FID reference distribution.

Outputs:
  prompts/coco_prompts.txt        50 captions, one per line
  data/coco_reference/            500 images at 512x512, indexed 0000.png .. 0499.png

Usage:
  python scripts/prepare_coco_data.py
"""

import io
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image

OUT_PROMPTS = Path("prompts/coco_prompts.txt")
OUT_REF_DIR = Path("data/coco_reference")
N_PROMPTS = 50
N_REFERENCE = 500
SEED = 42
TARGET_RES = 512
MAX_WORKERS = 16


def fetch_and_save(url: str, out_path: Path) -> bool:
    if out_path.exists():
        return True
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img = img.resize((TARGET_RES, TARGET_RES), Image.LANCZOS)
        img.save(out_path)
        return True
    except Exception as exc:
        print(f"  failed {url}: {exc}")
        return False


def main():
    OUT_PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_REF_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading phiyodr/coco2017 (validation split, parquet) ...")
    ds = load_dataset("phiyodr/coco2017", split="validation")
    print(f"  Loaded {len(ds)} (image, captions) rows")

    random.seed(SEED)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    prompt_indices = indices[:N_PROMPTS]
    reference_indices = indices[:N_REFERENCE]

    print(f"Writing {N_PROMPTS} captions to {OUT_PROMPTS} ...")
    with open(OUT_PROMPTS, "w", encoding="utf-8") as f:
        for idx in prompt_indices:
            row = ds[idx]
            caption = row["captions"][0].strip().replace("\n", " ")
            f.write(caption + "\n")

    print(f"Downloading {N_REFERENCE} reference images to {OUT_REF_DIR} ...")
    tasks = []
    for out_idx, ds_idx in enumerate(reference_indices):
        row = ds[ds_idx]
        url = row["coco_url"]
        out_path = OUT_REF_DIR / f"{out_idx:04d}.png"
        tasks.append((url, out_path))

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_and_save, u, p): (u, p) for u, p in tasks}
        for fut in as_completed(futures):
            ok = fut.result()
            completed += 1
            if completed % 50 == 0:
                print(f"  {completed}/{len(tasks)} downloaded")

    n_saved = len(list(OUT_REF_DIR.glob("*.png")))
    print(f"Done. {n_saved} reference images saved.")


if __name__ == "__main__":
    main()
