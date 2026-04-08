'''
given image (persona dataset)
extract dino features
dino features to SAE features
at inference, find layer, replace its activations with SAE features put through decoder or sum or average them

OR train tensors that we add to the SAEs instead of dreamboothing it  -that should be somewhere else

'''

import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
import torch

import time
import torch.nn.functional as F
import numpy as np
from loading import get_sae_dict
from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init
from experiment_helpers.argprint import print_args
from unet_autopsy import get_shape_dict
from overcomplete import TopKSAE
from dino_extract import dino_model,dino_processor,get_last_hidden_states
from PIL import Image
from data_helpers import PersonaDataset
from hook_wrapper import HookPipe
from diffusers import DiffusionPipeline
from eval_pcs import CLIPEvaluator
import wandb

parser=default_parser()
parser.add_argument("--layers",nargs="*",default=["down_blocks.1.attentions.1","down_blocks.2.attentions.1"])
parser.add_argument("--hidden_dim",nargs='*',help=" hidden dim of sae, if len = 1 then we default to all of them being the one thing")
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7")
parser.add_argument("--nb_concepts",type=int,default=10000,help="n concepts for SAE")
parser.add_argument("--parent_dir",type=str,default="sae_model")
parser.add_argument("--prefix",type=str,default="seg_ip_txt_")
parser.add_argument("--monkey",action="store_true",help="use monkey to generate images")
parser.add_argument("--step",type=int,default=2)
parser.add_argument("--subset",type=str,default="subject",help="subject or object or face")
parser.add_argument("--size",type=int,default=256)
parser.add_argument("--num_inference_steps",type=int,default=8)
#TODO: add monkey patching
            
#use persona dataset


        
        

def main(args):
    api,accelerator,device=repo_api_init(args)
    shape_dict=get_shape_dict(args.checkpoint,device,args.size)
    
    sae_dict=get_sae_dict(args.checkpoint,device,args.nb_concepts,args.layers,args.prefix,args.step)
    for layer,ksae in sae_dict.items():
        if torch.cuda.is_available():
            sae_dict[layer]=ksae.to(device)
    img = Image.new("RGB", (512, 512), color=(255, 255, 255))
    
    dino=get_last_hidden_states(img,dino_processor,dino_model).to(device)
    print("dino size ",dino.size())
    (b,n,dc)=dino.size()
    
    
    dino_sae_dict={
        layer: TopKSAE(dc,args.nb_concepts,device=device) for layer in args.layers
    }
    
    step=str(args.step)
    
    for layer,ksae in dino_sae_dict.items():
        if torch.cuda.is_available():
            ksae.load_state_dict(
                torch.load(
                    os.path.join("sae_model",args.prefix,layer,step,"dino_weights.pt")
                )
            )
        else:
            ksae.load_state_dict(
                torch.load(
                    os.path.join("sae_model",args.prefix,layer,step,"dino_weights.pt"),map_location=torch.device("cpu")
                )
            )
    #https://github.com/zhangxulu1996/awesome-personalization
    
    data=PersonaDataset(args.subset,(args.size,args.size),keyword=False)
    
    pipe=HookPipe(
        DiffusionPipeline.from_pretrained(args.checkpoint).to(device),args.layers,sae_dict,shape_dict,0.5
    )
    
    clip_text_alignment=[]
    clip_image_alignment=[]
    
    evaluator=CLIPEvaluator(device)
    
    with torch.no_grad():
    
        for r,row in enumerate(data):
            if r==args.limit:
                break
            
            image=row["image"]
            prompt=row["text"]
            keyword=row["keyword"]
            
            dino=get_last_hidden_states(image,dino_processor,dino_model)[:, 0, :].to(device)
            sae_src_dict={
                layer: sae_dict[layer].decode(ksae.encode(dino)[1]) for layer in dino_sae_dict
            }
                
            result=pipe.forward(sae_src_dict,prompt,num_inference_steps=args.num_inference_steps,height=256,width=256,return_dict=True,output_type="pil").images[0]
            
            accelerator.log({
                f"img_{r}":wandb.Image(result),
                keyword:wandb.Image(result)
            })
            
            clip_image_alignment.append(evaluator.img_to_img_similarity(result,row["image_pil"]).cpu().detach().numpy())
            clip_text_alignment.append(evaluator.txt_to_img_similarity(result,prompt).cpu().detach().numpy())
        
    
    print(np.mean(clip_text_alignment))
    print(np.mean(clip_image_alignment))
        
        
        
    

        


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")