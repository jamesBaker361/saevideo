import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
import torch

import time
import torch.nn.functional as F

from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init
from custom_pipeline_cosmos_text2world import CustomCosmosTextToWorldPipeline
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler,TextToVideoSDPipeline
from diffusers import HunyuanVideo15Pipeline
from diffusers import MochiPipeline
from hook_wrapper import HookWrapper
from torch.utils.data import Dataset,DataLoader
from datasets import Dataset,load_dataset
import numpy as np

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self,prompt_list):
        super().__init__()
        self.prompt_list=prompt_list
        
    def __len__(self):
        return len(self.prompt_list)
    
    def __getitem__(self, index):
        return self.prompt_list[index]
    
class OpenvidDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.src_data=load_dataset("nkp37/OpenVid-1M")["train"]
        
    def __len__(self):
        return len(self.src_data["caption"])
    
    def __getitem__(self, index):
        return self.src_data["caption"][index]

ALI="ali"
COSMOS="cosmos"
HUN="hun"
MOCHI="mochi"

parser=default_parser({"src_dataset":"nkp37/OpenVid-1M",})

parser.add_argument("--model",type=str,default=COSMOS)
parser.add_argument("--config",type=str,default="default")
parser.add_argument("--prompt_txt",type=str,default="video_prompts.txt")
parser.add_argument("--height",type=int,default=32)
parser.add_argument("--width",type=int,default=32)
parser.add_argument("--num_inference_steps",type=int,default=10)
parser.add_argument("--num_frames",type=int,default=5)
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/video-sae-test")
parser.add_argument("--save_interval",type=int,default=100)
            

def main(args):
    with torch.no_grad():
        
        folder=f"activations-{args.model}"
        os.makedirs(folder,exist_ok=True)
        config="config.csv"
        if os.path.exists(os.path.join(folder,config)):
            with open(os.path.join(folder,config),"r") as rf:
                count=len(rf.readlines())
        else:
            count=0
        print("count ",count)
        with open(os.path.join(folder,config),"a") as config_write_file:
            api,accelerator,device=repo_api_init(args)
            config_path=os.path.join("layer_dir",f"{args.model}_{args.config}.txt")
            with open(config_path,"r") as conf_file:
                model_layers=[c.strip() for c in conf_file.readlines()]
                
            kwargs={
                "num_inference_steps":args.num_inference_steps,
                "num_frames":args.num_frames,
                "height":args.height,
                "width":args.width,
            }
            
            dtype=torch.float16
            if args.model==COSMOS:
                model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
                pipe = CustomCosmosTextToWorldPipeline.from_pretrained(model_id, torch_dtype=dtype)
            elif args.model==HUN:
                pipe = HunyuanVideo15Pipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",torch_dtype=dtype)
            elif args.model==MOCHI:
                pipe=MochiPipeline.from_pretrained("genmo/mochi-1-preview",torch_dtype=dtype)
            elif args.model==ALI:
                pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=dtype, variant="fp16")
                
            pipe=pipe.to(device)
                
                
            hooked_pipe=HookWrapper(pipe,model_layers)
            
            with open(args.prompt_txt,"r") as prompt_file:
                prompt_list=[c.strip() for c in prompt_file.readlines()]
                
            dataset=OpenvidDataset()
            data=DataLoader(dataset,batch_size=args.batch_size)
            
            print("data len ",len(data))
            
            output_dict={
                "prompt":[]
            }
            for layer in model_layers:
                for step in range(args.num_inference_steps):
                    output_dict[f"{layer}_{step}"]=[]
                    
            
            
            for b,batch in enumerate(data):
                if b*args.batch_size<count:
                    continue
                if b==args.limit:
                    break
                video,act=hooked_pipe(batch, **kwargs)
                
                output_dict["prompt"]+=batch
                for layer in model_layers:
                    for step in range(args.num_inference_steps):
                        output_dict[f"{layer}_{step}"].extend(act[layer][step].numpy())
                for z in range(args.batch_size):
                    local_dict={}
                    for layer in model_layers:
                        for step in range(args.num_inference_steps):
                            local_dict[f"{layer}_{step}"]=act[layer][step][z].numpy()
                    save_path=os.path.join(folder,f"{z}.npz")
                    np.savez(save_path,**local_dict)
                    config_write_file.write(f"{batch[z]},{save_path}\n")
                            
                if b%args.save_interval==0:
                    Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)
                    
                        
            
            Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)        
            


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")