from diffusers import DiffusionPipeline
from hook_wrapper import HookWrapper
import argparse
import accelerate
import torch
import json
from diffusers.utils.loading_utils import load_image

def get_module_by_name(model, target_name):
    for name, module in model.named_modules():
        if name == target_name:
            return module
    raise ValueError(f"Module {target_name} not found")

def get_shape_dict(checkpoint:str,device,size:int=64,**kwargs)->dict[str,list[int]]:
    output_data={}
    try:
        pipe =DiffusionPipeline.from_pretrained(checkpoint).to(device,dtype=torch.float16)
    except torch.OutOfMemoryError:
        print("oom on device pipe on cpu")
        pipe=DiffusionPipeline.from_pretrained(checkpoint).to(dtype=torch.float32) #cpu doesnt support half precision iirc
    unet=pipe.unet
    names=[name for name,module in unet.named_modules()]

    hw=HookWrapper(pipe,names)
    
    _,act=hw("image",num_inference_steps=2, height=size,width=size,**kwargs)

    for k,v in act.items():
        output_data[k]=[t for t in v[0].size()]
        
    return output_data

def main():

    accelerator =accelerate.Accelerator()

    output_data={}

    for checkpoint in ["Lykon/dreamshaper-7","stabilityai/stable-diffusion-xl-base-1.0","stablediffusionapi/realistic-vision-v51","stabilityai/sdxl-turbo","SimianLuo/LCM_Dreamshaper_v7"]:
        print("\n\n\n\n")
        print(checkpoint)
        pipe=DiffusionPipeline.from_pretrained(checkpoint)
        shape_dict=get_shape_dict(checkpoint,accelerator.device)
        output_data[checkpoint]=shape_dict
        for key in shape_dict:
            print(key, type(get_module_by_name(pipe.unet,key)),shape_dict[key])
        
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    print("\n\n\n\n")
    print(checkpoint+ " h94/IP-Adapter")
    ip_adapter_image=load_image("https://www.aaha.org/wp-content/uploads/2024/09/kitten-lying-in-blanket.jpg")
    shape_dict=get_shape_dict(checkpoint,accelerator.device,size=64,ip_adapter_image=ip_adapter_image)
    output_data[checkpoint]=shape_dict
    for key in shape_dict:
        print(key, type(get_module_by_name(pipe.unet,key)),shape_dict[key])
    with open("autopsy.json","w") as file:
        json.dump(output_data,file)

if __name__=="__main__":        
    main()