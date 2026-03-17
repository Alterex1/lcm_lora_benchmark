import subprocess

MODELS = [
    "sd15_lcm",
    "sdxl_lcm",
    "ddim_sd15",
    "ddim_sdxl",
    "sdxl_turbo"
]

STEPS = [1, 2, 4, 8]

PROMPT_FILE = "prompts/prompts.txt"
MAX_PROMPTS = 10


def run_benchmark(model, steps):
    cmd = [
        "python",
        "scripts/benchmark_latency.py",
        "--model", model,
        "--steps", str(steps),
        "--prompt_file", PROMPT_FILE,
        "--max_prompts", str(MAX_PROMPTS),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


for model in MODELS:

    # Turbo usually doesn't support many steps
    if model == "sdxl_turbo":
        valid_steps = [1, 2, 4]
    else:
        valid_steps = STEPS

    for step in valid_steps:
        run_benchmark(model, step)

print("All benchmarks complete.")