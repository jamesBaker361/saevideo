import os
import subprocess

def install(package):
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", package])

with open("requirements.txt","r") as file:
    for line in file.readlines():
        install(line.strip())

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from cache_attn import CACHED_ATTN_WEIGHTS,insert_cache_attn
import math
from overcomplete.sae import TopKSAE
import huggingface_hub
import os
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from experiment_helpers.gpu_details import print_details
import argparse
from data_helpers import PersonaDataset
from eval_pcs import CLIPEvaluator
import numpy as np
from datasets import Dataset



dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
dino_model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

def get_last_hidden_states(image,dino_processor,dino_model)->torch.Tensor:
    inputs = dino_processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = dino_model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states


device = "cuda" if torch.cuda.is_available() else "cpu"



def get_mask(module:torch.nn.Module,
             n_tokens:int,
             threshold:float)->torch.Tensor:
    #print("monkey type",type(module))
    attn_weights=getattr(module,CACHED_ATTN_WEIGHTS)
    #assume attn weights are B,Heads,qdim,kdim
    #print("attn weights size",attn_weights.size())
    attn_weights=attn_weights.mean(dim=1).squeeze(0)
    #print("attn weights size", 42 ,attn_weights.size())
    #b, qdim kdim
    attn_weights=attn_weights[:,1:1+n_tokens]
    avg=attn_weights.mean(-1)
    #print('\tprocessor_kv[step].size()',processor_kv[step].size())
    #print("attn weights size",attn_weights.size())
    #print("\t avg ", avg.size())
    latent_dim=int (math.sqrt(avg.size()[0]))
    #print("\tlatent",latent_dim)
    avg=avg.view([latent_dim,latent_dim,-1])
    #print("\t avg ", avg.size())
    #print("\t avg ", avg.size())
    avg_min,avg_max=avg.min(),avg.max()
    x_norm = (avg - avg_min) / (avg_max - avg_min)  # [0,1]
    x_norm[x_norm < threshold]=0.
    x_norm[x_norm>0]=1.
    return x_norm

pipe = DiffusionPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7"
).to(device)

pipe.safety_checker = None
setattr(pipe,"safety_checker",None)
insert_cache_attn(pipe)

shape_dict={
     "down_blocks.1.attentions.0":640,
                "down_blocks.2.attentions.0":1280,
                "down_blocks.1.attentions.1":640,
                "down_blocks.2.attentions.1":1280,
                "mid_block.attentions.0":1280,
                "up_blocks.1.attentions.0":1280,
                "up_blocks.1.attentions.1":1280,
}
nb_concepts=10_000
sae_dict={
        block : TopKSAE(c,nb_concepts,device=device) for block,c in shape_dict.items()
    }
dc=384
nb_concepts=10000
dino_sae_dict={
    block : TopKSAE(dc,nb_concepts,device=device) for block in shape_dict
}

for block in shape_dict:
    weights_path=huggingface_hub.hf_hub_download(f"jlbaker361/sae_{block}",filename="weights.pt")

    # load weights
    if torch.cuda.is_available():
        sae_dict[block].load_state_dict(torch.load(weights_path))
    else:
        sae_dict[block].load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))
    sae_dict[block]=sae_dict[block].to(device)

    try:
        dino_weights_path=huggingface_hub.hf_hub_download(f"jlbaker361/sae_{block}",filename="dino_weights.pt")
        if os.path.exists(dino_weights_path):
            if torch.cuda.is_available():
                dino_sae_dict[block].load_state_dict(torch.load(dino_weights_path))
            else:
                dino_sae_dict[block].load_state_dict(torch.load(dino_weights_path,map_location=torch.device('cpu')))
            dino_sae_dict[block]=dino_sae_dict[block].to(device)
    except:
        pass


CACHED_ACTIVATIONS="cached_activations"
CACHED_OUTPUTS="cached_outputs"
SAVED_SAE="saved_sae"
INFERENCE_COUNTER="inference_step_counter"
CACHED_N_TOKENS="cached_n_tokens"
START="start_steps"
FINAL="final_steps"
WEIGHT="weight_sae"
THRESHOLD="threshold_mask"
weight=0.5
num_inference_steps=8
start_step=5
final_step=2
module_dict={}

def hook(module,input, output):
    #TODO: mask?
    steps=getattr(module,INFERENCE_COUNTER)
    steps-=1
    setattr(module,INFERENCE_COUNTER,steps)
    if steps> getattr(module,START) or steps<getattr(module,FINAL):
        return output
    if type(output)==tuple:
        dims=output[0].size()
    else:
        dims=output.size()
        
    monkey=module.transformer_blocks[0].attn2
    n_tokens=getattr(module,CACHED_N_TOKENS)
    mask=get_mask(monkey,n_tokens, getattr(module,THRESHOLD) ).squeeze(-1)
    activations=getattr(module,CACHED_ACTIVATIONS)
    activations=activations.unsqueeze(-1).unsqueeze(-1).expand(* dims)
    mask*=getattr(module,WEIGHT)
    mask=mask.to(device)
    
    if type(output)==tuple:
        out=(1-mask)*output[0] + mask*(activations+input[0])
        if len(output)==1:
            return (out,)
        else:
            return (out, * output[1:])
    else:
        return (1-mask)*output + mask*(activations+input[0])
for layer,mod in pipe.unet.named_modules():
    if layer in shape_dict:
        setattr(mod,INFERENCE_COUNTER,num_inference_steps)
        setattr(mod,START,start_step)
        setattr(mod,FINAL,final_step)
        setattr(mod,WEIGHT,0.5)
        module_dict[layer]=mod
        mod.register_forward_hook(hook)

@torch.inference_mode()
def generate(
    prompt:str,
    keyword:str,
    size:int,
    num_inference_steps:int,
    src_image,
    weight:float,threshold:float,
    start_step:int,
    final_step
)->Image.Image:
    full_prompt = f"{keyword} {prompt}"
    dino=get_last_hidden_states(src_image,dino_processor,dino_model)[:, 0, :].to(device)
    n_tokens=len(pipe.tokenizer.encode(keyword))-2 #excluding the first and last start/end tokens
    for layer in shape_dict:
        activations=sae_dict[layer].decode(dino_sae_dict[layer].encode(dino)[1])
        setattr(module_dict[layer],CACHED_ACTIVATIONS,activations)
        setattr(module_dict[layer],INFERENCE_COUNTER,num_inference_steps)
        setattr(module_dict[layer],CACHED_N_TOKENS,n_tokens)
        setattr(module_dict[layer],START,start_step)
        setattr(module_dict[layer],FINAL,final_step)
        setattr(module_dict[layer],WEIGHT,weight)
        setattr(module_dict[layer],THRESHOLD,threshold)
    image = pipe(
        full_prompt,
        num_inference_steps=num_inference_steps,
        height=size,
        width=size
    ).images[0]

    return image


if __name__=="__main__":
    print_details()
    parser=argparse.ArgumentParser()
    parser.add_argument("--start_step",type=int,default=6)
    parser.add_argument("--final_step",type=int,default=2)
    parser.add_argument("--size",type=int,default=256)
    parser.add_argument("--weight",type=float,default=0.75)
    parser.add_argument("--threshold",type=float,default=0.75)
    parser.add_argument("--name",type=str,default="demo")
    
    args=parser.parse_args()
    
    data=PersonaDataset("subject", (256,256))
    evaluator=CLIPEvaluator(device)
    clip_text_alignment=[]
    clip_image_alignment=[]
    hf_data={"src":[],"prompt":[],"result":[]}
    with torch.no_grad():
    
        for r,row in enumerate(data):
            
            image=row["image"]
            prompt=row["text"]
            keyword=row["keyword"]
                
            dino=get_last_hidden_states(image,dino_processor,dino_model)[:, 0, :].to(device)
            n_tokens=len(pipe.tokenizer.encode(keyword))-2 #excluding the first and last start/end tokens
            for layer in shape_dict:
                activations=sae_dict[layer].decode(dino_sae_dict[layer].encode(dino)[1])
                setattr(module_dict[layer],CACHED_ACTIVATIONS,activations)
                setattr(module_dict[layer],INFERENCE_COUNTER,num_inference_steps)
                setattr(module_dict[layer],CACHED_N_TOKENS,n_tokens)
                
            result=generate(prompt,keyword,args.size,8,image,args.weight,args.threshold,args.start_step,args.final_step)
            
            clip_image_alignment.append(evaluator.img_to_img_similarity(result,row["image_pil"]).cpu().detach().numpy())
            clip_text_alignment.append(evaluator.txt_to_img_similarity(result,prompt).cpu().detach().numpy())
                
            hf_data["prompt"].append(prompt)
            hf_data["result"].append(result)
            hf_data["src"].append(image)
            
        print(np.mean(clip_text_alignment))
        print(np.mean(clip_image_alignment))
    
        Dataset.from_dict(hf_data).push_to_hub(f"jlbaker361/{args.name}")