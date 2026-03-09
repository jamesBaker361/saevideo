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

class PromptDataset(Dataset):
    def __init__(self,prompt_list):
        super().__init__()
        self.prompt_list=prompt_list
        
    def __len__(self):
        return len(self.prompt_list)
    
    def __getitem__(self, index):
        return self.prompt_list[index]

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
            

def main(args):
    with torch.no_grad():
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
            
            
        hooked_pipe=HookWrapper(pipe,model_layers)
        
        with open(args.prompt_txt,"r") as prompt_file:
            prompt_list=[c.strip() for c in prompt_file.readlines()]
            
        dataset=PromptDataset(prompt_list)
        data=DataLoader(dataset,batch_size=args.batch_size)
        
        output_dict={
            "prompt":[]
        }
        for layer in model_layers:
            output_dict[layer]=[]
        
        for batch in data:
            video,act=hooked_pipe(batch, **kwargs)
            


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