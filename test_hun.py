import torch

dtype = torch.bfloat16

from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator

from experiment_helpers.gpu_details import print_details
print_details()


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = HunyuanVideo15Pipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)

pipe.vae.enable_tiling()

seed = 123
prompt = "xyz"

generator = torch.Generator(device=device).manual_seed(seed)

with torch.no_grad():
    video = pipe(
        prompt=prompt,
        generator=generator,
        num_frames=20,
        num_inference_steps=20,
        width=64,
        height=48
    ).frames[0]

export_to_video(video, "output.mp4", fps=24)

print("all done!")