#this is to just extract and collect the features of a normal UNet (no ip adapter)
from diffusers import DiffusionPipeline
from hook_wrapper import HookWrapper
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.init_helpers import repo_api_init,default_parser,parse_args
import time
from datasets import load_dataset
import os
import numpy as np
import torch

DREAMSHAPER="dreamshaper"
SDXL="sdxl"
REALISTIC="realistic"

layer_dict={
    DREAMSHAPER:[ #this is actually probably what you should do for both not sdxl things?
        "conv_in",
        "down_blocks.0.attentions.0","down_blocks.0.attentions.1","down_blocks.0.downsamplers",
        "down_blocks.1.attentions.0","down_blocks.1.attentions.1","down_blocks.1.downsamplers",
        "down_blocks.2.attentions.0","down_blocks.2.attentions.1","down_blocks.2.downsamplers",
        "down_blocks.3.resnets.0","down_blocks.3.resnets.1",
        "up_blocks.0.upsamplers","up_blocks.0.resnets.0","up_blocks.0.resnets.1","up_blocks.0.resnets.2",
        "up_blocks.1.upsamplers","up_blocks.1.attentions.0","up_blocks.1.attentions.1",
        "up_blocks.2.attentions.0","up_blocks.2.attentions.1","up_blocks.2.upsamplers",
        "up_blocks.3.attentions.0","up_blocks.3.attentions.1","up_blocks.3.upsamplers",
        "mid_block.attentions.0","conv_out"
    ],
    SDXL:["conv_in","down_blocks.0.downsamplers",
          "down_blocks.1.attentions.0","down_blocks.1.attentions.1","down_blocks.1.downsamplers",
          "down_blocks.2.attentions.0","down_blocks.2.attentions.1",
          "up_blocks.0.attentions.0","up_blocks.0.attentions.1","up_blocks.0.attentions.2","up_blocks.0.upsamplers",
          "up_blocks.1.attentions.0","up_blocks.1.attentions.1","up_blocks.1.attentions.2","up_blocks.1.upsamplers",
          "mid_block.attentions.0",
          "conv_out"
          ],
    REALISTIC:["down_blocks.0.attentions.0","down_blocks.0.attentions.1","down_blocks.0.downsamplers",
               "down_blocks.1.attentions.0","down_blocks.1.attentions.1","down_blocks.1.downsamplers",
               "down_blocks.2.attentions.0","down_blocks.2.attentions.1","down_blocks.2.downsamplers"]
}

parser=default_parser({"save_dir":"feature_dir"})
parser.add_argument("--pipeline",type=str,default="SimianLuo/LCM_Dreamshaper_v7")
parser.add_argument("--num_inference_steps",type=int,default=4)

parser.add_argument("--data",type=str,default="AnyModal/flickr30k")
parser.add_argument("--size",type=int,default=256)
parser.add_argument("--layer_set",type=str,default=DREAMSHAPER,help="")

def main(args):
    api,accelerator,device=repo_api_init(args)
    is_cpu = device.type == "cpu"
    dtype = torch.float32 if is_cpu else torch.float16
    pipe=DiffusionPipeline.from_pretrained(args.pipeline,torch_dtype=dtype)
    pipe.to(device)
    pipe.enable_model_cpu_offload()
    layers=layer_dict[args.layer_set]
    hw=HookWrapper(pipe,layers)
    
    data=load_dataset(args.data,split="train")
    
    os.makedirs(args.save_dir,mode=0o777,exist_ok=True)
    
    for n in range(args.num_inference_steps):
        os.makedirs(os.path.join(args.save_dir,str(n)),exist_ok=True)
        
    os.makedirs(os.path.join(args.save_dir,"images"),exist_ok=True)
        
    count=len([f for f in os.listdir(os.path.join(args.save_dir,str(n))) if f.endswith("npz")])
    
    print("count ",count)
    
    text_key="alt_text"
    
    if args.data=="guangyil/laion-coco-aesthetic":
        text_key="TEXT"
    with torch.no_grad():
        for r,row in enumerate(data):
            if r<count:
                continue
            if r==args.limit:
                break
            img,act=hw(row[text_key][0],height=args.size,width=args.size,num_inference_steps=args.num_inference_steps) 
            img_path=os.path.join(args.save_dir,"images",f"{r}.png")
            img.save(img_path)
            for n in range(args.num_inference_steps):
                results={k:v[n] for k,v in act.items()}
                path=os.path.join(args.save_dir,str(n),f"act_{r}.npz")
                np.savez(path,**results)
    

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parse_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")