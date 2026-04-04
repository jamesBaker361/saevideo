from diffusers import DiffusionPipeline
from hook_wrapper import HookWrapper
import argparse
import accelerate
import torch
import json



def get_shape_dict(checkpoint:str,device,size:int=64)->dict[str,list[int]]:
    output_data={}
    pipe =DiffusionPipeline.from_pretrained(checkpoint).to(device,dtype=torch.float16)
    unet=pipe.unet
    names=[name for name,module in unet.named_modules()]

    hw=HookWrapper(pipe,names)
    
    _,act=hw("image",num_inference_steps=2, height=size,width=size)

    for k,v in act.items():
        output_data[k]=[t for t in v[0].size()]
        
    return output_data

def main():

    accelerator =accelerate.Accelerator()

    output_data={}

    for checkpoint in ["SimianLuo/LCM_Dreamshaper_v7","Lykon/dreamshaper-7","stabilityai/stable-diffusion-xl-base-1.0","stablediffusionapi/realistic-vision-v51"]:
        print("\n\n\n\n")
        print(checkpoint)
        pipe=DiffusionPipeline.from_pretrained(checkpoint)
        shape_dict=get_shape_dict(checkpoint,accelerator.device)
        output_data[checkpoint]=shape_dict
        for key in shape_dict:
            print(key, type(getattr(pipe.unet,key,None)),shape_dict[key])
        
    with open("autopsy.json","w") as file:
        json.dump(output_data,file)

if __name__=="__main__":        
    main()