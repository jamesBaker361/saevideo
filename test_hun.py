import torch

dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video
from experiment_helpers.gpu_details import print_details
print_details()

pipe = HunyuanVideo15Pipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", )
import accelerate
accelerator =accelerate.Accelerator()
pipe=accelerator.prepare(pipe)

pipe.vae.enable_tiling()

seed=123
prompt="xyz"
generator = torch.Generator(device=device).manual_seed(seed)

video = pipe(
    prompt=prompt,
    generator=generator,
    num_frames=121,
    num_inference_steps=50,
    width=64,
    height=48
).frames[0]

export_to_video(video, "output.mp4", fps=24)

print("all done!")