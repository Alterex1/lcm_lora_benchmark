import torch
from diffusers import (
    AutoPipelineForText2Image,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
)


def _detect_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    elif torch.cuda.is_available():
        return "cuda", torch.float16
    else:
        return "cpu", torch.float32


def load_pipeline(model_name: str):
    device, dtype = _detect_device_and_dtype()
    variant = "fp16" if dtype == torch.float16 else None

    pretrained_kwargs = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if variant:
        pretrained_kwargs["variant"] = variant

    if model_name == "sd15_lcm":
        print("Loading Stable Diffusion 1.5 + LCM-LoRA...")
        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            **pretrained_kwargs,
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

    elif model_name == "sdxl_lcm":
        print("Loading SDXL + LCM-LoRA...")
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            **pretrained_kwargs,
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

    elif model_name == "ddim_sd15":
        print("Loading Stable Diffusion 1.5 + DDIM...")
        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            **pretrained_kwargs,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    elif model_name == "ddim_sdxl":
        print("Loading SDXL + DDIM...")
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            **pretrained_kwargs,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    elif model_name == "sdxl_turbo":
        print("Loading SDXL-Turbo...")
        turbo_kwargs = {"torch_dtype": dtype}
        if variant:
            turbo_kwargs["variant"] = variant
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            **turbo_kwargs,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe
