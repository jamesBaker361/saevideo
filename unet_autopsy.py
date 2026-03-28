from diffusers import DiffusionPipeline
from hook_wrapper import HookWrapper
import argparse
import accelerate
import json

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7")

args=parser.parse_args()

accelerator =accelerate.Accelerator()

output_data={}

for checkpoint in ["SimianLuo/LCM_Dreamshaper_v7","Lykon/dreamshaper-7","stabilityai/stable-diffusion-xl-base-1.0","stablediffusionapi/realistic-vision-v51"]:
    print("\n\n\n\n")
    output_data[checkpoint]={}

    pipe =DiffusionPipeline.from_pretrained(checkpoint).to(accelerator.device)

    print(type(pipe))
    print(dir(pipe))
    unet=pipe.unet
    for name,module in unet.named_modules():
        print(name,type(module))
        
    names=[name for name,module in unet.named_modules()]

    hw=HookWrapper(pipe,names)

    _,act=hw("image",num_inference_steps=12, height=256,width=256)

    for k,v in act.items():
        print(k,v[0].size())
        output_data[checkpoint][k]=[t for t in v[0].size()]
        
with open("autopsy.json","w") as file:
    json.dump(output_data,file)