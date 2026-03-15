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

parser=default_parser()
parser.add_argument("--pipeline",type=str,default="SimianLuo/LCM_Dreamshaper_v7")
parser.add_argument("--inference_steps",type=int,default=4)

parser.add_argument("--data",type=str,default="AnyModal/flickr30k")
parser.add_argument("--size",type=int,default=256)

def main(args):
    api,accelerator,device=repo_api_init(args)
    pipe=DiffusionPipeline.from_pretrained(args.pipeline)
    pipe.to(device)
    layers=[
        "down_blocks.0.attentions.0","down_blocks.0.attentions.1","down_blocks.0.downsamplers",
        "down_blocks.1.attentions.0","down_blocks.1.attentions.1","down_blocks.1.downsamplers",
        "down_blocks.2.attentions.0","down_blocks.2.attentions.1","down_blocks.2.downsamplers",
        "down_blocks.3.resnets.0","down_blocks.3.resnets.1",
        "up_blocks.0.upsamplers","up_blocks.0.resnets.0","up_blocks.0.resnets.1","up_blocks.0.resnets.2",
        "up_blocks.1.upsamplers","up_blocks.1.attentions.0","up_blocks.1.attentions.1",
        "up_blocks.2.attentions.0","up_blocks.2.attentions.1","up_blocks.2.upsamplers",
        "up_blocks.3.attentions.0","up_blocks.3.attentions.1","up_blocks.3.upsamplers",
        "mid_block.attentions.0"
    ]
    hw=HookWrapper(pipe,layers)
    
    data=load_dataset(args.data,split="train")
    
    os.makedirs(args.save_dir,True)
    
    for n in range(args.num_inference_steps):
        os.makedirs(os.path.join(args.save_dir,n),exist_ok=True)
        
    count=len([f for f in os.listdir(os.path.join(args.save_dir,n)) if f.endswith("npz")])
    
    print("count ",count)
    
    for r,row in data:
        if r<count:
            continue
        if r==args.limit:
            break
        img,act=hw(row["alt_text"][0],height=args.size,width=args.size,num_inference_steps=args.num_inference_steps)
        for n in range(args.num_inference_steps):
            results={k:v[n] for k,v in act.items()}
            path=os.path.join(args.save_dir,n,f"act_{n}.npz")
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