#https://github.com/surkovv/sdxl-unbox

from diffusers import DiffusionPipeline
from hook_wrapper import HookPipe,HookUNet
from overcomplete import TopKSAE
import torch
import numpy as np
import os
from accelerate import Accelerator
from copy import deepcopy
from diffusers.utils.loading_utils import load_image
from experiment_helpers.argprint import print_args
import json
from PIL import Image
from loading import get_sae_dict


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
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import DataLoader,Dataset
from unet_autopsy import get_shape_dict


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

class DreamboothDataset(Dataset):
    def __init__(self,key:str,text_encoder,tokenizer):
        super().__init__()
        self.image_processor=VaeImageProcessor()
        self.tokenizer=tokenizer
        self.text_encoder=text_encoder

        
        unique_token="<sks>"
        samples_per_epoch=128 #this has to be high in case we want large batch sizes
        
        with open("pcs_dataset/info.json","r") as f:
            mapping=json.load(f)
            
        
        if key in mapping["subjects"]["subject_with_cls"]:
            category="subjects"
            class_token =mapping["subjects"]["subject_with_cls"][key]
        elif key in mapping["face"]["id_with_gender"]:
            category="face"
            class_token= mapping["face"]["id_with_gender"][key]

        self.prompt_list = [
                    'a {0} {1} '.format(unique_token, class_token) for _ in range(samples_per_epoch)
            ]
            
        self.image_list=[]
        for n in range(samples_per_epoch):
            try:
                if category=="face":
                    img=Image.open(os.path.join("pcs_dataset",category,key,"face.jpg"))
                elif category=="subjects":
                    img=Image.open(os.path.join("pcs_dataset",category,key,f"0{n}.jpg"))
                self.image_list.append(img)
            except:
                break
            
    def __len__(self):
        return len(self.prompt_list)
    
    def __getitem__(self, index):
        return {
            "image":self.image_processor.preprocess(self.image_list[index %len(self.image_list)]),
            "input_ids":compute_text_embeddings(self.prompt_list[index],self.tokenizer,self.text_encoder)
        }
        
                  
        
        
        

parser=default_parser()

parser.add_argument("--weight",type=float,default=0.5)
parser.add_argument("--key",type=str,default="chair")
parser.add_argument("--checkpoint",type=str,default="stablediffusionapi/realistic-vision-v51")
parser.add_argument("--num_inference_steps",type=int,default=2)
parser.add_argument("--nb_concepts",type=int,default=10000,help="n concepts for SAE")
parser.add_argument("--prefix",type=str,default="features_stablediffusionapi_realistic-vision-v51_32_")
parser.add_argument("--step",type=int,default=24)



def main(args):
    api,accelerator,device=repo_api_init(args)
    os.makedirs(args.save_dir,exist_ok=True)


    pipe=DiffusionPipeline.from_pretrained(args.checkpoint,torch_dtype=torch.float32).to(device)

    layers=[
        "down_blocks.1.attentions.0","down_blocks.1.attentions.1",
                        "down_blocks.2.attentions.0","down_blocks.2.attentions.1",
                        "up_blocks.1.attentions.0","up_blocks.1.attentions.1",
                        "up_blocks.2.attentions.1",#"up_blocks.2.upsamplers",
                        "up_blocks.3.attentions.1",#"up_blocks.3.upsamplers",
                        "mid_block.attentions.0"
    ]
    nb_concepts=args.nb_concepts
        
    def get_unet_device_dtype(unet):
        param = next(unet.parameters())
        return param.device, param.dtype

    device, dtype = get_unet_device_dtype(pipe.unet)
    shape_dict=get_shape_dict(args.checkpoint,device,64)
    sae_dict=get_sae_dict(args.checkpoint,device,args.nb_concepts,layers,args.prefix,args.step)



    vae=pipe.vae
    text_encoder=pipe.text_encoder
    unet=pipe.unet
    scheduler=pipe.scheduler
    tokenizer=pipe.tokenizer
    
    for model in [vae,text_encoder,unet]:
        model.requires_grad_(False)
    
    hooked_unet=HookUNet(unet,layers,sae_dict,args.weight)

    batch_size=1
    #maybe do one of these for each entity (which )
    sae_src_dict={
            lay: torch.nn.Parameter(torch.randn(1, nb_concepts, device=device))
            for lay in layers
        }#these are trainable!
    
    
    params=[v for v in sae_src_dict.values()]
    optimizer_class = torch.optim.AdamW
    optimizer=optimizer_class(params,args.lr)
    
    
    unique_token="<sks>"
    class_token="dog"
    
    data=DreamboothDataset(args.key,text_encoder,tokenizer)
    
    start_epoch=1
    for epoch in range(start_epoch, args.epochs+1):
        loss_list=[]
        for row in data:
            img=row["image"].to(device)
            encoder_hidden_states=row["input_ids"]
            latents=vae.encode(img).latent_dist.sample()
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
                    model_pred = hooked_unet.forward( #TODO: add cross attn matching attn2.to_k
                        sae_src_dict,
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
    for lay,t in sae_src_dict.items():
        new_dir=os.path.join(args.save_dir,lay)
        os.makedirs(new_dir,exist_ok=True)
        new_path=os.path.join(new_dir,"weights.pt")
        torch.save(t,new_path)
        
    #log result images
    with open("pcs_dataset/info.json","r") as f:
        mapping=json.load(f)
        
    if args.key in mapping["subjects"]["subject_with_cls"]:
        category="subjects"
        class_token =mapping["subjects"]["subject_with_cls"][args.key]
        
        if class_token in mapping["subjects"]["live_subjects"]:
            prompt_list= mapping["subjects"]["prompt_live"]
        else:
            prompt_list=mapping["subjects"]["prompt_object"]
    elif args.key in mapping["face"]["id_with_gender"]:
        category="face"
        class_token= mapping["face"]["id_with_gender"][args.key]
        prompt_list = [p for subset in ["prompt_accesory","prompt_context","prompt_action","prompt_style"] for p in mapping[category][subset]]
        
    prompt_list=[p.format(unique_token,class_token) for p in prompt_list]
    
    hooked_pipeline=HookPipe(DiffusionPipeline.from_pretrained(args.checkpoint,torch_dtype=torch.float16).to(device),
                             layers,sae_dict,shape_dict,args.weight)
    
    for p,prompt in enumerate(prompt_list):
        gen_img:Image.Image=hooked_pipeline.forward(sae_src_dict,prompt,height=256,width=256,num_inference_steps=args.num_inference_steps).images[0]
        gen_img.save(os.path.join(args.save_dir,f"gen_{p}.jpg"))
            

    #hooker.forward(sae_src_list,"walking",height=64,width=64,num_inference_steps=4)
    
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