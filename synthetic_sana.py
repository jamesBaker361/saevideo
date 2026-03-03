import os
from accelerate import Accelerator
from datasets import Dataset

subject_path="subjects.txt"
style_path="styles.txt"

with open(subject_path,"r") as subject_file:
    subject_list=subject_file.readlines()
    
with open(style_path, "r") as style_file:
    style_list=style_file.readlines()
    
accelerator=Accelerator()
device=accelerator.device

print(device)

src_dict={
    "image":[],
    "subject":[],
    "style":[]
}

import torch
from diffusers import StableDiffusion3Pipeline
import time

from diffusers import SanaSprintPipeline
import torch

pipe = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16
)


pipe = pipe.to(device)

start=time.time()
limit=30

for b,sub in enumerate(subject_list):
    for y,sty in enumerate(style_list):
        
        if b*100 +y ==limit:
            end=time.time()
            print(f"total time = {end-start}, aka {(end-start)/(b*100 +y)}")
        elif b+100+y >limit:
            break
        else:

            image = pipe(
                f"{sub} {sty}",
                num_inference_steps=2,
                #guidance_scale=4.5,
            ).images[0]
            
            src_dict["image"].append(image)
            src_dict["style"].append(sty)
            src_dict["subject"].append(sub)

Dataset.from_dict(src_dict).push_to_hub("jlbaker361/synthetic-sana")