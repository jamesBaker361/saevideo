#uses ip adapter images to generate features of stuff

import os
#os.environ["TQDM_DISABLE"] = "1"
import sys
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from accelerate import Accelerator
import time

from diffusers import UNet2DConditionModel
import torch.nn.functional as F
import math
sys.path.append(os.path.dirname(__file__))
from ipattn import MonkeyIPAttnProcessor, get_modules_of_types,reset_monkey,insert_monkey, set_ip_adapter_scale_monkey
import torch
from experiment_helpers.image_helpers import concat_images_horizontally
from PIL import Image
from compatible_pipelines import CompatibleLatentConsistencyModelPipeline
from hook_wrapper import HookWrapper
from dino_extract import dino_model, dino_processor, get_last_hidden_states


#from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector

import datasets
from datasets import Dataset
import numpy as np

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="seg-ip-sae")
parser.add_argument("--load_hf",action="store_true",help="whether to load a special pretrained model")
parser.add_argument("--embedding",type=str, help="ignore unless load from hf; its the embedding type for embedding helpers")
parser.add_argument("--pretrained_model_path",type=str,default="")
parser.add_argument("--src_dataset",type=str, default="jlbaker361/synthetic-sana")
parser.add_argument("--use_test_split",action="store_true", help="only true for league dataset")
parser.add_argument("--initial_steps",type=int,default=4,help="how many steps for the initial inference")
parser.add_argument("--initial_mask_step_list",nargs="*",help="steps to generate mask from",type=int,default=[1,2])
parser.add_argument("--threshold",type=float,default=0.5,help="threshold for mask")
parser.add_argument("--limit",type=int,default=-1,help="limit of samples")
parser.add_argument("--layer_index",type=int,default=15)
parser.add_argument("--dim",type=int,default=256)
parser.add_argument("--token",type=int,default=1, help="which IP token is attention")
parser.add_argument("--overlap_frac",type=float,default=0.8)
parser.add_argument("--segmentation_attention_method",type=str,help="overlap or exclusive",default="overlap")
parser.add_argument("--kv_type",type=str,default="ip")
parser.add_argument("--initial_ip_adapter_scale",type=float,default=0.75)
parser.add_argument("--background",action="store_true")
parser.add_argument("--dest_dataset",type=str, default="jlbaker361/monkey-sae")
parser.add_argument("--object",type=str,default="character")
parser.add_argument("--save_dir",type=str,default="seg_ip")
parser.add_argument("--src_dir",type=str,default="synthetic_sana_synth_txt")
parser.add_argument("--hf_data",action="store_true")

'''
TODO:
> add hooks for diffusion model
> add hooks to generated dataset 


Per this papaer: https://arxiv.org/pdf/2504.15473 
down_blocks.2.attentions.1, mid_block.attentions.0, up_blocks.1.attentions.0
'''

def get_mask(processor_kv:list[torch.Tensor],
             step:int,
             token:int,
             threshold:float):
    
    avg=processor_kv[step].mean(dim=1).squeeze(0)
    #print("\t avg ", avg.size())
    latent_dim=int (math.sqrt(avg.size()[0]))
    #print("\tlatent",latent_dim)
    avg=avg.view([latent_dim,latent_dim,-1])
    #print("\t avg ", avg.size())
    avg=avg[:,:,token]
    #print("\t avg ", avg.size())
    avg_min,avg_max=avg.min(),avg.max()
    x_norm = (avg - avg_min) / (avg_max - avg_min)  # [0,1]
    x_norm[x_norm < threshold]=0.
    #avg = (x_norm * 255)
    #avg=F.interpolate(avg.unsqueeze(0).unsqueeze(0), size=(dim, dim), mode="nearest").squeeze(0).squeeze(0)

    return avg


def main(args):
    with torch.no_grad():
        prompt_txt="real_test_prompt_list.txt"
        if os.path.exists(prompt_txt):
            with open(prompt_txt,"r") as file:
                real_test_prompt_list=[s.strip() for s in file.readlines()]
        else:
            real_test_prompt_list=["in space","on a bus"]
        
        accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
        accelerator.init_trackers(project_name=args.project_name,config=vars(args))



        if args.initial_mask_step_list is None:
            initial_quarter=args.initial_steps //4
            args.initial_mask_step_list=[f for f in range(args.initial_steps)][initial_quarter:-initial_quarter]
            accelerator.print("defaulting to initial_mask_step_list",args.initial_mask_step_list )


        pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float16,
        ).to(accelerator.device)
        
        unet:UNet2DConditionModel =pipe.unet

        # Load IP-Adapter
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        setattr(pipe,"safety_checker",None)
        
        
        def hook(module,input,output):
            setattr(module,"cached_input",input)
            if type(input)==tuple:
                for i,input_tensor in enumerate(input):
                    if torch.isnan(input_tensor).any():
                        print("nan input ",i)
            setattr(module,"cached_output",output)
            if type(output)==tuple:
                for o,output_tensor in enumerate(output):
                    if torch.isnan(output_tensor).any():
                        print("nan output ",o)
            return output
        
        block_list=[
            "down_blocks.0.attentions.0",
            "down_blocks.1.attentions.0",
            "down_blocks.2.attentions.0",
            "down_blocks.0.attentions.1",
            "down_blocks.1.attentions.1",
            "down_blocks.2.attentions.1",
            "mid_block.attentions.0",
            "up_blocks.1.attentions.0",
            "up_blocks.2.attentions.0",
            "up_blocks.1.attentions.1",
            "up_blocks.2.attentions.1",
        ]
        block_dict={}
        for name,module in unet.named_modules():
            if name in block_list:
                module.register_forward_hook(hook)
                block_dict[name]=module
                
        
        set_ip_adapter_scale_monkey(pipe,args.initial_ip_adapter_scale)
        insert_monkey(pipe)
        print("len pipe.unet.attn_processors ",len(pipe.unet.attn_processors))
        for name,processor in pipe.unet.attn_processors.items():
            if hasattr(processor,"kv_ip"):
                print(f"{name} is type {type(processor)} and has property kv_ip")

        #monkey_attn_list=get_modules_of_types(pipe.unet,MonkeyIPAttnProcessor)
        
        if args.hf_data:
            try:
                data=datasets.load_dataset(args.src_dataset)
            except:
                data=datasets.load_dataset(args.src_dataset,download_mode="force_redownload")
            data=data["train"]
        else:
            data=[{"path":file} for file in os.listdir(args.src_dir) if (file.endswith("png") or file.endswith("jpg"))][:args.limit]
            #data=[{"image":Image.open(os.path.join(args.src_dir,file)) }for file in path_list]
                
        os.makedirs(args.save_dir,mode=0o777,exist_ok=True)
    
        for n in range(args.initial_steps):
            os.makedirs(os.path.join(args.save_dir,str(n)),exist_ok=True)
            
        for key in ["image","mask","mask_int"]:
            os.makedirs(os.path.join(args.save_dir,key),exist_ok=True)
            
        os.makedirs(os.path.join(args.save_dir,"dino"),exist_ok=True)
        
        count=len([f for f in os.listdir(args.save_dir) if f.endswith("npz")])
        
        print(f"found {count} different images total data len {len(data)}")

        for k,row in enumerate(data):
            if k<count:
                continue
            
            new_path=os.path.join(args.save_dir,f"{k}.npz")
            if os.path.exists(new_path):
                continue
            if k==args.limit:
                break
            reset_monkey(pipe)
            if "image" in row:
                ip_adapter_image=row["image"]
            else:
                ip_adapter_image=Image.open(os.path.join(args.src_dir,row["path"]))
            
            dino=get_last_hidden_states(ip_adapter_image,dino_processor,dino_model)
            
            object=args.object
            if "object" in row:
                object=row["object"]
            prompt=object+real_test_prompt_list[k % len(real_test_prompt_list)]

            generator=torch.Generator()
            generator.manual_seed(123)
            set_ip_adapter_scale_monkey(pipe,0.5)
            initial_image=pipe(prompt,args.dim,args.dim,args.initial_steps,ip_adapter_image=ip_adapter_image,generator=generator).images[0]
            activation_dict={
                "dino":dino.cpu().detach().numpy()
            }
            for block,module in block_dict.items():
                for key in ["input","output"]:
                    activation=getattr(module,f"cached_{key}")
                    if type(activation)==tuple:
                        activation=activation[0]
                    activation_dict[f"{key}.{block}"]=activation.cpu().detach().numpy()
                    setattr(module,f"cached_{key}",None)
            for name,processor in pipe.unet.attn_processors.items():
                if hasattr(processor,"kv_ip"):
                    mask=sum([get_mask(processor.kv_ip,step, args.token,args.threshold) for step in args.initial_mask_step_list])
                    if torch.isnan(mask).any():
                        print("nan mask")
                    mask=mask.cpu().detach().numpy()
                    activation_dict[f"mask.{name}"]=mask
                    #setattr(processor,"kv_ip",None)
            np.savez(new_path,**activation_dict)
            initial_image.save(os.path.join(args.save_dir,f"{k}.jpg"))
            if k==count:
                print("saving to ",os.path.join(args.save_dir,f"{k}.npz"),os.path.join(args.save_dir,f"{k}.jpg"))
            accelerator.free_memory()


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    print_args(parser)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")