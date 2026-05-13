import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.argprint import print_args
import torch

import time
import torch.nn.functional as F
from diffusers import DiffusionPipeline,UNet2DConditionModel,AutoencoderKL

from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init
from data_helpers import PersonaDataset
from eval_pcs import CLIPEvaluator
from unet_autopsy import get_shape_dict
from overcomplete.sae import TopKSAE,QSAE, JumpSAE, BatchTopKSAE,losses,SAE
from dino_extract import dino_model,dino_processor,get_last_hidden_states
from PIL import Image
from accelerate import Accelerator
import wandb
import numpy as np
from ipattn import reset_monkey,insert_monkey
import math

def get_unet_device_dtype(unet:UNet2DConditionModel):
    param = next(unet.parameters())
    return param.device, param.dtype

def get_mask(monkey:torch.nn.Module,
             n_tokens:int,
             kv_type:str,
             step:int,
             threshold:float):
    print("monkey type",type(monkey))
    if kv_type=="ip":
        processor_kv=monkey.processor.kv_ip
    elif kv_type=="str":
        processor_kv=monkey.kv
    #print('\tprocessor_kv[step].size()',processor_kv[step].size())
    
    avg=processor_kv[step].mean(dim=1).squeeze(0)
    #print("\t avg ", avg.size())
    latent_dim=int (math.sqrt(avg.size()[0]))
    #print("\tlatent",latent_dim)
    avg=avg.view([latent_dim,latent_dim,-1])
    #print("\t avg ", avg.size())
    avg=avg[:,:,1:1+n_tokens].means(-1)
    print("\t avg ", avg.size())
    avg_min,avg_max=avg.min(),avg.max()
    x_norm = (avg - avg_min) / (avg_max - avg_min)  # [0,1]
    x_norm[x_norm < threshold]=0.
    x_norm[x_norm>0]=1.
    return x_norm
    
def generate(device,
             size:int,
             nb_concepts:int,
             block_list:list[str],
             dir_list:list[str],
             num_inference_steps:int,
             start_step:int,
             final_step:int,
             weight:float,
             accelerator:Accelerator,
             subset:str,
             limit:int=-1,
             checkpoint:str="SimianLuo/LCM_Dreamshaper_v7"):
    
    pipe=DiffusionPipeline.from_pretrained(checkpoint).to(device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    setattr(pipe,"safety_checker",None)
    insert_monkey(pipe)
    reset_monkey(pipe)
    data=PersonaDataset(subset,(size,size),keyword=False)
    evaluator=CLIPEvaluator(device)
    
    assert len(block_list)==len(dir_list), "len(block_list)!=len(dir_list)"
    shape_dict=get_shape_dict(checkpoint,device,size)
    sae_dict={
        block : TopKSAE(shape_dict[block][1],nb_concepts,device=device) for block in block_list
    }
    img = Image.new("RGB", (512, 512), color=(255, 255, 255))
    dino=get_last_hidden_states(img,dino_processor,dino_model).to(device)
    print("dino size ",dino.size())
    (b,n,dc)=dino.size()
    dino_sae_dict={
        block : TopKSAE(dc,nb_concepts,device=device) for block in block_list
    }
    unet:UNet2DConditionModel=pipe.unet
    device,dtype=get_unet_device_dtype(unet)
    for block,load_dir in zip(block_list,dir_list):
        weights_path = os.path.join(load_dir, "weights.pt")
        config_path = os.path.join(load_dir, "config.json")
        # load weights
        sae_dict[block].load_state_dict(torch.load(weights_path))
        sae_dict[block]=sae_dict[block].to(device)


        dino_weights_path = os.path.join(load_dir, "dino_weights.pt")
        if os.path.exists(dino_weights_path):
            dino_sae_dict[block].load_state_dict(torch.load(dino_weights_path))
            dino_sae_dict[block]=dino_sae_dict[block].to(device)
                
    CACHED_ACTIVATIONS="cached_activations"
    CACHED_OUTPUTS="cached_outputs"
    SAVED_SAE="saved_sae"
    INFERENCE_COUNTER="inference_step_counter"
    CACHED_N_TOKENS="cached_n_tokens"
    module_dict={}
    for layer,mod in unet.named_modules():
        if layer in block_list:
            setattr(mod,INFERENCE_COUNTER,num_inference_steps)
            module_dict[layer]=mod
            def hook(module,input, output):
                #TODO: mask?
                steps=getattr(module,INFERENCE_COUNTER)
                steps-=1
                setattr(module,INFERENCE_COUNTER,steps)
                if steps> start_step or steps<final_step:
                    return output
                if type(output)==tuple:
                    dims=output[0].size()
                else:
                    dims=output.size()
                    
                monkey=module.transformer_blocks[0].attn2
                n_tokens=getattr(module,CACHED_N_TOKENS)
                mask=get_mask(monkey,1,"str",-1,0.5)
                activations=getattr(module,CACHED_ACTIVATIONS)
                activations=activations.unsqueeze(-1).unsqueeze(-1).expand(* dims)
                mask*=weight
                mask=mask.to(device,dtype)
                
                if type(output)==tuple:
                    out=(1-mask)*output[0] + mask*activations
                    if len(output)==1:
                        return (out,)
                    else:
                        return (out, * output[1:])
                else:
                    return (1-mask)*output + mask*activations
                
            mod.register_forward_hook(hook)
    
    clip_text_alignment=[]
    clip_image_alignment=[]
    
    evaluator=CLIPEvaluator(device)
    
    with torch.no_grad():
    
        for r,row in enumerate(data):
            if r==limit:
                break
            
            image=row["image"]
            prompt=row["text"]
            keyword=row["keyword"]
                
            dino=get_last_hidden_states(image,dino_processor,dino_model)[:, 0, :].to(device)
            n_tokens=len(pipe.tokenizer.encode(keyword))-2 #excluding the first and last start/end tokens
            for layer in block_list:
                activations=sae_dict[layer].decode(dino_sae_dict[layer].encode(dino)[1])
                setattr(module_dict[layer],CACHED_ACTIVATIONS,activations)
                setattr(module_dict[layer],INFERENCE_COUNTER,num_inference_steps)
                setattr(module_dict[layer],CACHED_N_TOKENS,n_tokens)
                
            result=pipe(keyword+" "+prompt,num_inference_steps=num_inference_steps,height=size,width=size,return_dict=True,output_type="pil").images[0]
            
            accelerator.log({
                f"img_{r}":wandb.Image(result),
                keyword:wandb.Image(result)
            })
            
            clip_image_alignment.append(evaluator.img_to_img_similarity(result,row["image_pil"]).cpu().detach().numpy())
            clip_text_alignment.append(evaluator.txt_to_img_similarity(result,prompt).cpu().detach().numpy())
                
                
            
        print(np.mean(clip_text_alignment))
        print(np.mean(clip_image_alignment))
    
    

parser=default_parser()
parser.add_argument("--size",type=int,default=256)
parser.add_argument("--nb_concepts",type=int,default=10000)
parser.add_argument("--subset",type=str,default="subject",help="subject or object or face")

def main(args):
    size:int=args.size
    subset:str=args.subset
    limit:int=args.limit
    api,accelerator,device=repo_api_init(args)
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
    dir_list=["sae_"+b for b in block_list]
    generate(
        device,args.size,args.nb_concepts,block_list,dir_list,8,5,2,0.5,accelerator,subset,limit
    )     


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