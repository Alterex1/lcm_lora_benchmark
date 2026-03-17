import torch
from diffusers import DiffusionPipeline


def load_pipeline(model_name: str):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if model_name == "sd15_test":
        print("Loading Stable Diffusion 1.5 test model...")

        dtype = torch.float32
        if device == "cuda":
            dtype = torch.float16

        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        pipe = pipe.to(device)
        pipe.enable_attention_slicing()

        return pipe

    raise ValueError(f"Unknown model name: {model_name}")