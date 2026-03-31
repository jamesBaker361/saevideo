#https://github.com/surkovv/sdxl-unbox

from diffusers import DiffusionPipeline
from hook_wrapper import HookForward,HookUNet
from overcomplete import TopKSAE
import torch
import numpy as np
import os
from accelerate import Accelerator
from copy import deepcopy


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

parser=default_parser()

parser.add_argument("--weight",type=float,default=0.5)

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds
            
            
def compute_text_embeddings(prompt,tokenizer,text_encoder):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=77)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                )

            return prompt_embeds

def main(args):
    api,accelerator,device=repo_api_init(args)
    os.makedirs(args.save_dir,exist_ok=True)


    pipe=DiffusionPipeline.from_pretrained("stablediffusionapi/realistic-vision-v51",torch_dtype=torch.float16).to(device)

    layers=[
        "down_blocks.1.attentions.0","down_blocks.1.attentions.1",
                        "down_blocks.2.attentions.0","down_blocks.2.attentions.1",
                        "up_blocks.1.attentions.0","up_blocks.1.attentions.1",
                        "up_blocks.2.attentions.1",#"up_blocks.2.upsamplers",
                        "up_blocks.3.attentions.1",#"up_blocks.3.upsamplers",
                        "mid_block.attentions.0"
    ]
    nb_concepts=10000
    c_list=[]
    step=24
    src_dir=f"features_stablediffusionapi_realistic-vision-v51_32"
    lay_dict={}
    for lay in layers:
        try:
            raw_activations=torch.tensor(np.load(os.path.join(src_dir, "0","0.npz"))[lay][0])
        except:
            for key in np.load(os.path.join(src_dir, "0","act_0.npz")).keys():
                raw_activations=torch.tensor(np.load(os.path.join(src_dir, "0","act_0.npz"))[lay][0])
        act_size=raw_activations.size()
        (c,h,w)=act_size
        c_list.append(c)
        print(lay,act_size)
        lay_dict[lay]=c
        
    def get_unet_device_dtype(unet):
        param = next(unet.parameters())
        return param.device, param.dtype

    device, dtype = get_unet_device_dtype(pipe.unet)
    sae_dict={
        lay: TopKSAE(c,nb_concepts,).to(device,dtype) for lay,c in lay_dict.items()
    }



    for lay,ksae in sae_dict.items():
        ksae.load_state_dict(torch.load(
                os.path.join("sae_model",f"{src_dir}_{lay}_{step}","weights.pt")
                ))
        ksae.requires_grad_(False)



    vae=pipe.vae
    text_encoder=pipe.text_encoder
    unet=pipe.unet
    scheduler=pipe.scheduler
    tokenizer=pipe.tokenizer
    
    for model in [vae,text_encoder,unet]:
        model.requires_grad_(False)
    
    hooked_unet=HookUNet(unet,layers,sae_dict,args.weight)
    blank_unet=deepcopy(unet)

    batch_size=1
    #maybe do one of these for each entity (which )
    sae_src_list={ lay: torch.normal(0,1,(batch_size,nb_concepts)).to(device,dtype)  for lay in layers} #these are trainable!
    optimizer_class = torch.optim.AdamW
    
    params=[v for v in sae_src_list.values()]
    optimizer=optimizer_class(params,args.lr)
    
    data=[]
    
    start_epoch=1
    for epoch in range(start_epoch, args.num_train_epochs+1):
        loss_list=[]
        for row in data:
            img=row["image"]
            encoder_hidden_states=row["input_ids"]
            latents=vae.encode(img).latent_dict.sample()
            noise = torch.randn_like(latents)
            
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (batch_size,), device=latents.device
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = scheduler.add_noise(latents, noise, timesteps)
            

            
            
            with accelerator.accumulate(params):
                with accelerator.autocast():
                    optimizer.zero_grad()
                    model_pred = unet(
                        noisy_model_input, timesteps, encoder_hidden_states, class_labels=None, return_dict=False
                    )[0]
                    
                    if scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif scheduler.config.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(latents, noise, timesteps)
                        
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    accelerator.backward(loss)
                    optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            
        accelerator.log({
            "loss":np.mean(loss_list)
        })
        print("loss",np.mean(loss_list))
    os.makedirs(args.save_dir,exist_ok=True)
    for lay,t in sae_src_list:
        new_dir=os.path.join(args.save_dir,lay)
        os.makedirs(new_dir,exist_ok=True)
        new_path=os.path.join(new_dir,"weights.pt")
        torch.save(t,new_path)                
        

    #hooker.forward(sae_src_list,"walking",height=64,width=64,num_inference_steps=4)