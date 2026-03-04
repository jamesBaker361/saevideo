import os
import sys
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time

import torch.nn.functional as F
import math
from diffusers.models.attention_processor import  IPAdapterAttnProcessor2_0,Attention
from diffusers.image_processor import IPAdapterMaskProcessor
sys.path.append(os.path.dirname(__file__))
from ipattn import MonkeyIPAttnProcessor, get_modules_of_types,reset_monkey,insert_monkey, set_ip_adapter_scale_monkey
import torch
from experiment_helpers.image_helpers import concat_images_horizontally
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, CLIPModel
from compatible_pipelines import CompatibleLatentConsistencyModelPipeline
#import ImageReward as RM



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
parser.add_argument("--src_dataset",type=str, default="jlbaker361/ssl-league_captioned_splash-1000-sana")
parser.add_argument("--use_test_split",action="store_true", help="only true for league dataset")
parser.add_argument("--initial_steps",type=int,default=4,help="how many steps for the initial inference")
parser.add_argument("--initial_mask_step_list",nargs="*",help="steps to generate mask from",type=int,default=[1,2])
parser.add_argument("--final_steps",type=int,default=8, help="how many steps for final inference (with mask)")
parser.add_argument("--final_mask_steps_list",nargs="*",help="steps to apply mask from",type=int)
parser.add_argument("--final_adapter_steps_list",nargs="*",help="steps to apply adapter for (regardless of mask)",type=int)
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

def get_mask(layer_index:int, 
             attn_list:list,step:int,
             token:int,dim:int,
             threshold:float,
             kv_type:str="ip",
             vae_scale:int=8):
    #print("layer",layer_index)
    module=attn_list[layer_index][1] #get the module no name
    #module.processor.kv_ip
    if kv_type=="ip":
        processor_kv=module.processor.kv_ip
    elif kv_type=="str":
        processor_kv=module.processor.kv
    size=processor_kv[step].size()
    #print('\tprocessor_kv[step].size()',processor_kv[step].size())
    
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
    avg = (x_norm * 255)
    #avg=F.interpolate(avg.unsqueeze(0).unsqueeze(0), size=(dim, dim), mode="nearest").squeeze(0).squeeze(0)

    return avg

class ScoreTracker:
    def __init__(self):
        self.score_list_dict={
                "dino_score_unmasked":[],
                "dino_score_seg_mask":[],
                "dino_score_raw_mask":[],
                "dino_score_normal":[],
                "dino_score_all_steps":[],
                "text_score_unmasked":[],
                "text_score_seg_mask":[],
                "text_score_raw_mask":[],
                "text_score_normal":[],
                "text_score_all_steps":[],
                "image_score_unmasked":[],
                "image_score_seg_mask":[],
                "image_score_raw_mask":[],
                "image_score_normal":[],
                "image_score_all_steps":[],
            }

    def update(self,score_dict):
        for k,v in score_dict.items():
            self.score_list_dict[k].append(v)

    def get_means(self)-> dict:
        ret={}
        for k,v in self.score_list_dict.items():
            if len(v)>0:
                ret[k]=np.mean(v)

        return ret

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
        if args.final_mask_steps_list is None:
            final_quarter=args.final_steps //4
            args.final_mask_steps_list=[f for f in range(args.final_steps)][final_quarter:-final_quarter]
            accelerator.print("defaulting final maske step lst",args.final_mask_steps_list )
        if args.final_adapter_steps_list is None:
            args.final_adapter_steps_list=args.final_mask_steps_list


        pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float16,
        ).to(accelerator.device)

        # Load IP-Adapter
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        set_ip_adapter_scale_monkey(pipe,args.initial_ip_adapter_scale)

        setattr(pipe,"safety_checker",None)

        insert_monkey(pipe)
        attn_list=get_modules_of_types(pipe.unet,Attention)

        #monkey_attn_list=get_modules_of_types(pipe.unet,MonkeyIPAttnProcessor)
        try:
            data=datasets.load_dataset(args.src_dataset)
        except:
            data=datasets.load_dataset(args.src_dataset,download_mode="force_redownload")
        data=data["train"]

        

        if args.background:
            background_data=datasets.load_dataset("jlbaker361/real_test_prompt_list",split="train")
            background_dict={row["prompt"]:row["image"] for row in background_data}
            accelerator.print("background dict", background_dict)

        score_tracker=ScoreTracker()
        if args.background:
            background_score_tracker=ScoreTracker()

        output_dict={
        "image":[],
        "mask":[],
        "mask_int":[]
        }

        for k,row in enumerate(data):
            if k==args.limit:
                break
            reset_monkey(pipe)
            ip_adapter_image=row["image"]
            object=args.object
            if "object" in row:
                object=row["object"]
            prompt=object+real_test_prompt_list[k % len(real_test_prompt_list)]
            if args.background:
                background_image=background_dict[prompt.replace(object,"")]
                prompt=" "
            generator=torch.Generator()
            generator.manual_seed(123)
            set_ip_adapter_scale_monkey(pipe,0.5)
            accelerator.print("inital image")
            initial_image=pipe(prompt,args.dim,args.dim,args.initial_steps,ip_adapter_image=ip_adapter_image,generator=generator).images[0]

            mask=sum([get_mask(args.layer_index,attn_list,step,args.token,args.dim,args.threshold) for step in args.initial_mask_step_list])
            tiny_mask=mask.clone()
            tiny_mask_pil=to_pil_image(1-tiny_mask)
            #print("mask size",mask.size())

            mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(args.dim, args.dim), mode="nearest").squeeze(0).squeeze(0)

            

            mask_pil=to_pil_image(1-mask)
            color_rgba = initial_image.convert("RGB")
            mask_pil = mask_pil.convert("RGB")
            
            masked_img=Image.blend(color_rgba, mask_pil, 0.5)

            mask[mask>1]=1.
            inverted_mask=1.0-mask
            
            mask_int_pil=to_pil_image(mask)
            
            output_dict["image"].append(initial_image)
            output_dict["mask"].append(mask_pil) 
            output_dict["mask_int"].append(mask_int_pil)
            
            
        Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)
        accelerator.print("Average Scores:")
        accelerator.print(len(avg_score_dict))
        for k,v in avg_score_dict.items():
            accelerator.print(k,float(v))
        if args.background:
            avg_score_dict=background_score_tracker.get_means()

            accelerator.print("Background Average Scores:")
            accelerator.print(len(avg_score_dict))
            for k,v in avg_score_dict.items():
                accelerator.print(k,float(v))





        




    return

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