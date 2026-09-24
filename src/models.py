import torch
from pipelines.expansion_pipeline import StableDiffusionExpansionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def expansion_pipeline():
    pipe = StableDiffusionExpansionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.load_lora_weights("./resources/spex_lora", weight_name="pytorch_lora_weights.safetensors", adapter_name="pattern_lora")
    pipe.set_adapters(["pattern_lora"], adapter_weights=[0.95])
    pipe.enable_freeu(b1=1.1, b2=1.2, s1=0.6, s2=0.4)
    pipe = pipe.to(DEVICE)
    return pipe