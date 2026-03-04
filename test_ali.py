import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler,TextToVideoSDPipeline
from diffusers.utils import export_to_video
from accelerate import Accelerator

from experiment_helpers.gpu_details import print_details
print_details()

accelerator = Accelerator()
device = accelerator.device

pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#pipe.enable_model_cpu_offload()

pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

prompt = "Spiderman is surfing"
with torch.no_grad():
    video_frames = pipe(prompt, num_inference_steps=25,height=160,width=320,num_frames=10).frames
    
print(type(video_frames),type(pipe))
print(video_frames.shape)
video_path = export_to_video(video_frames[0],"ali.mp4")


print("all done!")