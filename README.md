# LCM-LoRA Benchmarking Project

## Overview

This project evaluates **Latent Consistency Models (LCM-LoRA)** for fast image generation and compares them against baseline diffusion methods:

* **DDIM sampling**
* **SDXL-Turbo**

We analyze the **trade-off between speed and quality** across different inference step counts.

---

## Objectives

* Compare image generation **latency**
* Evaluate **image quality** using:

  * CLIP Score (prompt alignment)
  * FID (distribution similarity)
* Analyze performance at **1, 2, 4, and 8 steps**

---

## Models Evaluated

| Model        | Description                        |
| ------------ | ---------------------------------- |
| `sd15_lcm`   | Stable Diffusion 1.5 with LCM-LoRA |
| `sdxl_lcm`   | Stable Diffusion XL with LCM-LoRA  |
| `ddim_sd15`  | SD1.5 with DDIM sampling           |
| `ddim_sdxl`  | SDXL with DDIM sampling            |
| `sdxl_turbo` | SDXL-Turbo (optimized fast model)  |

---

## Project Structure

```
lcm_lora_benchmark/

scripts/
  model_loader.py
  generate_images.py
  benchmark_latency.py
  compute_clip_score.py
  compute_fid.py
  run_all_benchmarks.py

prompts/
  sample_prompts.txt

results/
  metrics/
    latency_results.csv
    clip_scores.csv
    fid_scores.csv

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

```bash
pip install torch torchvision
pip install diffusers transformers accelerate safetensors pillow tqdm scipy peft numpy
```

---

## Running the Project

### 1. Generate Images

```bash
python scripts/generate_images.py \
  --prompt_file prompts/sample_prompts.txt \
  --max_prompts 10
```

---

### 2. Benchmark Latency

```bash
python scripts/benchmark_latency.py \
  --model sd15_lcm \
  --steps 4 \
  --prompt_file prompts/sample_prompts.txt \
  --max_prompts 10
```

---

### 3. Run All Benchmarks

```bash
python scripts/run_all_benchmarks.py
```

---

### 4. Compute CLIP Score

```bash
python scripts/compute_clip_score.py \
  --image_base_dir results/images \
  --prompt_file prompts/sample_prompts.txt \
  --max_prompts 10
```

---

### 5. Compute FID Score

Requires reference images:

```bash
python scripts/compute_fid.py \
  --image_base_dir results/images \
  --reference_dir data/reference_images
```

---

## Results

All results are stored in:

```
results/metrics/
```

### Metrics

* **Latency** → `latency_results.csv`
* **CLIP Score** → `clip_scores.csv`
* **FID Score** → `fid_scores.csv`

---

## Key Findings (Summary)

* **LCM-LoRA** achieves significantly faster inference at low step counts
* **DDIM** provides higher quality at higher steps but is slower
* **SDXL-Turbo** offers a strong speed/quality trade-off
* There is a clear **speed vs quality trade-off** across all methods

---

## Notes

* Generated images are **not included** in the repository to reduce size
* HuggingFace models are downloaded automatically at runtime
* For best performance, run experiments on a **GPU (CUDA-enabled machine)**

---

## Authors

* Ahatesham Bhuiyan
* Erick Castillo
* Aisha Fathalla
* Tiffany Petrovic

---

## License

For academic use only.
