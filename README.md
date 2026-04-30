# LCM-LoRA Benchmarking Project

## Overview

This project benchmarks **Latent Consistency Models (LCM-LoRA)** for fast text-to-image generation and compares them against two baselines:

* **DDIM sampling** on the same Stable Diffusion backbones
* **SDXL-Turbo** (adversarially distilled single-step generator)

We measure the **speed–quality trade-off** across $1, 2, 4, 8$ inference steps and additionally test the **LCM-LoRA universality claim** — that the acceleration adapter can be linearly composed with an arbitrary style/domain LoRA at inference time without retraining.

---

## Objectives

* Compare per-image **latency**, throughput, and peak GPU memory
* Evaluate **image quality** using:
  * **CLIP Score** (prompt alignment) on $50$ COCO 2017 validation captions
  * **FID** against a domain-matched $500$-image COCO reference set at $512^2$
* Analyze performance at **1, 2, 4, and 8 steps**
* Empirically verify the LCM-LoRA universality claim ($\tau' = \lambda_1 \tau_{\text{style}} + \lambda_2 \tau_{\text{LCM}}$) using a public pixel-art style LoRA

---

## Models Evaluated

| Model              | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `sd15_lcm`         | Stable Diffusion 1.5 with LCM-LoRA                         |
| `sdxl_lcm`         | Stable Diffusion XL with LCM-LoRA                          |
| `ddim_sd15`        | SD1.5 with DDIM sampling                                   |
| `ddim_sdxl`        | SDXL with DDIM sampling                                    |
| `sdxl_turbo`       | SDXL-Turbo (adversarially distilled fast model)            |
| `sdxl_styled`      | SDXL + pixel-art style LoRA, DDIM (universality reference) |
| `sdxl_styled_lcm`  | SDXL + pixel-art style LoRA + LCM-LoRA (universality test) |

---

## Project Structure

```
lcm_lora_benchmark/

scripts/
  model_loader.py                 model + adapter loader for all configs
  generate_images.py              main benchmark image generation
  benchmark_latency.py            per-image latency / throughput / memory
  compute_clip_score.py           CLIP ViT-L/14 prompt alignment
  compute_fid.py                  FID against a reference image set
  run_all_benchmarks.py           sweep latency over all (model, steps) cells
  prepare_coco_data.py            fetch 50 COCO captions + 500 reference images
  run_universality_experiment.py  LCM-LoRA + style-LoRA composition test
  build_qualitative_grid.py       assemble qualitative-comparison grid
  plot_results.py                 generate latency/CLIP/FID/Pareto plots

prompts/
  sample_prompts.txt              5 descriptive prompts (qualitative pilot)
  coco_prompts.txt                50 COCO 2017 val captions (main benchmark)

data/
  coco_reference/                 500 COCO val images at 512x512 (FID reference)

results/
  metrics/
    latency_results.csv
    clip_scores.csv               N=50 COCO results
    fid_scores.csv                COCO reference results
    clip_scores_n5_cifar.csv      archived original (N=5 prompts)
    fid_scores_n5_cifar.csv       archived original (CIFAR-10 reference)
    fid_scores_partial.csv        partial pre-HPC FID run, kept for posterity
  figures/
    latency_vs_steps.png
    clip_vs_steps.png
    fid_vs_steps.png
    pareto_clip_vs_latency.png
    qualitative_grid.png
    universality_grid.png

requirements.txt
README.md
.gitignore
```

---

## Setup

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.\.venv\Scripts\activate    # Windows
```

### 2. Install dependencies

Install `torch` first with the right CUDA version for your system, then everything else:

```bash
# CUDA 12.x (most modern GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older HPC modules)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU-only (FID-only on a CPU node)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

---

## Running the Project

### 1. Prepare COCO data (one-time)

Downloads $50$ captions to `prompts/coco_prompts.txt` and $500$ reference images to `data/coco_reference/`.

```bash
python scripts/prepare_coco_data.py
```

### 2. Generate Images

Sweeps every `(model, steps)` cell and saves to `results/images/<model>/steps_<N>/`.

```bash
python scripts/generate_images.py \
  --prompt_file prompts/coco_prompts.txt \
  --max_prompts 50
```

### 3. Benchmark Latency

Per (model, steps) cell:

```bash
python scripts/benchmark_latency.py \
  --model sd15_lcm \
  --steps 4 \
  --prompt_file prompts/coco_prompts.txt \
  --max_prompts 50
```

Or sweep all cells:

```bash
python scripts/run_all_benchmarks.py
```

### 4. Compute CLIP Score

```bash
python scripts/compute_clip_score.py \
  --image_base_dir results/images \
  --prompt_file prompts/coco_prompts.txt \
  --max_prompts 50
```

### 5. Compute FID

```bash
python scripts/compute_fid.py \
  --image_base_dir results/images \
  --reference_dir data/coco_reference \
  --output_csv results/metrics/fid_scores.csv
```

FID can run **CPU-only** if no GPU is available — useful for HPC CPU nodes.

### 6. Universality Experiment

Generates SDXL+style LoRA at $30$/$4$ DDIM steps and SDXL+style+LCM at $4$ LCM steps on the $5$-prompt qualitative pilot, then assembles the comparison grid.

```bash
python scripts/run_universality_experiment.py
```

### 7. Plots and Qualitative Grid

After running steps 2–5:

```bash
python scripts/plot_results.py            # latency / CLIP / FID / Pareto figures
python scripts/build_qualitative_grid.py  # 5x5 qualitative comparison grid
```

Output figures are saved to `results/figures/`.

---

## Results

All metrics are stored as CSVs in `results/metrics/`:

* **Latency** → `latency_results.csv`
* **CLIP Score** → `clip_scores.csv`
* **FID Score** → `fid_scores.csv`

Plots in `results/figures/` are regenerated from these CSVs via `scripts/plot_results.py`.

---

## Key Findings (Summary)

Quantitative results from the $50$-COCO-prompt benchmark with COCO-matched FID reference:

* **DDIM at 1 step is unusable.** CLIP $\approx 12$, FID $\geq 498$ for both backbones — visibly pure colored noise. DDIM only recovers to a competitive CLIP score by step $4$ for SD1.5 ($25.26$) and step $8$ for SDXL ($26.89$).
* **LCM-LoRA produces coherent images at $1$ step** (CLIP $22$–$24$, FID $287$–$300$) and is approximately flat from step $2$ onward (CLIP $\approx 26$, FID $\approx 240$). This directly reflects the self-consistency property targeted by LCM training.
* **SDXL-Turbo gives the best CLIP-per-second** at $512^2$: CLIP $26.72$, FID $244.87$ at $1$ step in $\sim\!0.35$ s/img on the reference hardware.
* **LCM-LoRA universality holds empirically.** Linearly composing the public LCM-LoRA-SDXL adapter with `nerijs/pixel-art-xl` at $\lambda_1 = 0.8, \lambda_2 = 1.0$ produces stylized $4$-step outputs that visually match the $30$-step reference, while the same style LoRA at $4$ steps without LCM produces incoherent outputs.

See the project report for the full analysis and figures.

---

## Notes

* Generated images and the COCO reference set are **not committed** to the repository (gitignored) to keep the clone size small. Re-run `scripts/prepare_coco_data.py` and `scripts/generate_images.py` to regenerate them locally; both are deterministic given fixed seeds.
* HuggingFace models (`runwayml/stable-diffusion-v1-5`, `stabilityai/stable-diffusion-xl-base-1.0`, `stabilityai/sdxl-turbo`, the LCM-LoRA adapters, the pixel-art LoRA) download automatically on first use. Total cache size is roughly $15$ GB.
* For full reproducibility, image generation and latency benchmarks need a CUDA-enabled GPU with $\geq 12$ GB VRAM in FP16. CLIP scoring and FID can run on CPU.

---

## Authors

* Ahatesham Bhuiyan
* Erick Castillo
* Aisha Fathalla
* Tiffany Petrovic

---

## License

For academic use only.
