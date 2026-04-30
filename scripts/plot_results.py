"""
Generate paper figures from the benchmark CSVs.

Outputs four PNGs to results/figures/:
  - latency_vs_steps.png
  - clip_vs_steps.png
  - fid_vs_steps.png
  - pareto_clip_vs_latency.png

Usage:
  python scripts/plot_results.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRICS_DIR = Path("results/metrics")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["sd15_lcm", "sdxl_lcm", "ddim_sd15", "ddim_sdxl", "sdxl_turbo"]
MODEL_STYLE = {
    "sd15_lcm":   {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "sdxl_lcm":   {"color": "#ff7f0e", "marker": "s", "linestyle": "-"},
    "ddim_sd15":  {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
    "ddim_sdxl":  {"color": "#d62728", "marker": "v", "linestyle": "--"},
    "sdxl_turbo": {"color": "#9467bd", "marker": "D", "linestyle": "-."},
}


def load_latency():
    df = pd.read_csv(METRICS_DIR / "latency_results.csv")
    # The CSV contains two runs; keep the first occurrence of each (model, steps).
    df = df.drop_duplicates(subset=["model_name", "steps"], keep="first")
    return df.sort_values(["model_name", "steps"])


def load_clip():
    return pd.read_csv(METRICS_DIR / "clip_scores.csv").sort_values(["model_name", "steps"])


def load_fid():
    return pd.read_csv(METRICS_DIR / "fid_scores.csv").sort_values(["model_name", "steps"])


def plot_latency(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in MODEL_ORDER:
        sub = df[df["model_name"] == model]
        if sub.empty:
            continue
        ax.plot(
            sub["steps"], sub["avg_latency_sec"],
            label=model, **MODEL_STYLE[model],
        )
    ax.set_xlabel("Inference steps")
    ax.set_ylabel("Avg latency per image (s)")
    ax.set_title("Latency vs. inference steps")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latency_vs_steps.png", dpi=200)
    plt.close(fig)


def plot_clip(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in MODEL_ORDER:
        sub = df[df["model_name"] == model]
        if sub.empty:
            continue
        ax.plot(
            sub["steps"], sub["avg_clip_score"],
            label=model, **MODEL_STYLE[model],
        )
    ax.set_xlabel("Inference steps")
    ax.set_ylabel("Avg CLIP score (higher is better)")
    ax.set_title("CLIP score vs. inference steps")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "clip_vs_steps.png", dpi=200)
    plt.close(fig)


def plot_fid(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in MODEL_ORDER:
        sub = df[df["model_name"] == model]
        if sub.empty:
            continue
        ax.plot(
            sub["steps"], sub["fid_score"],
            label=model, **MODEL_STYLE[model],
        )
    ax.set_xlabel("Inference steps")
    ax.set_ylabel("FID vs. CIFAR-10 reference (lower is better)")
    ax.set_title("FID vs. inference steps")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fid_vs_steps.png", dpi=200)
    plt.close(fig)


def plot_pareto(latency_df, clip_df):
    merged = latency_df.merge(
        clip_df[["model_name", "steps", "avg_clip_score"]],
        on=["model_name", "steps"],
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in MODEL_ORDER:
        sub = merged[merged["model_name"] == model]
        if sub.empty:
            continue
        ax.plot(
            sub["avg_latency_sec"], sub["avg_clip_score"],
            label=model, **MODEL_STYLE[model],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{int(row['steps'])}",
                (row["avg_latency_sec"], row["avg_clip_score"]),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
                color=MODEL_STYLE[model]["color"],
            )
    ax.set_xlabel("Avg latency per image (s)")
    ax.set_ylabel("Avg CLIP score")
    ax.set_title("Speed--quality trade-off (number = inference steps)")
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pareto_clip_vs_latency.png", dpi=200)
    plt.close(fig)


def main():
    latency_df = load_latency()
    clip_df = load_clip()
    fid_df = load_fid()

    plot_latency(latency_df)
    plot_clip(clip_df)
    plot_fid(fid_df)
    plot_pareto(latency_df, clip_df)

    print(f"Saved figures to {OUT_DIR.resolve()}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
