import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
from experiment_helpers.gpu_details import print_details
print_details()

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
frames = pipe(prompt, num_frames=84).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)


print("all done!")