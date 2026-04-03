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
from unet_autopsy import get_feature_dict


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
        live_subjects_map = {
            "cat": "cat",
            "cat2": "cat",
            "dog": "dog",
            "dog2": "dog",
            "dog3": "dog",
            "dog5": "dog",
            "dog6": "dog",
            "dog7": "dog",
            "dog8": "dog",
        }

        objects_map = {
            "backpack": "backpack",
            "backpack_dog": "backpack",
            "bear_plushie": "stuffed animal",
            "berry_bowl": "bowl",
            "can": "can",
            "candle": "candle",
            "clock": "clock",
            "colorful_sneaker": "sneaker",
            "duck_toy": "toy",
            "fancy_boot": "boot",
            "grey_sloth_plushie": "stuffed animal",
            "monster_toy": "toy",
            "pink_sunglasses": "glasses",
            "poop_emoji": "toy",
            "rc_car": "toy",
            "red_cartoon": "cartoon",
            "robot_toy": "toy",
            "shiny_sneaker": "sneaker",
            "teapot": "teapot",
            "vase": "vase",
            "wolf_plushie": "stuffed animal",
        }
        
        unique_token="<sks>"
        
        if key in objects_map:
            class_token=objects_map[key]
            self.prompt_list = [
            'a {0} {1} in the jungle'.format(unique_token, class_token),
            'a {0} {1} in the snow'.format(unique_token, class_token),
            'a {0} {1} on the beach'.format(unique_token, class_token),
            'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
            'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
            'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
            'a {0} {1} with a city in the background'.format(unique_token, class_token),
            'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
            'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
            'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
            'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
            'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
            'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
            'a {0} {1} floating on top of water'.format(unique_token, class_token),
            'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
            'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
            'a {0} {1} on top of a mirror'.format(unique_token, class_token),
            'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
            'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
            'a {0} {1} on top of a white rug'.format(unique_token, class_token),
            'a red {0} {1}'.format(unique_token, class_token),
            'a purple {0} {1}'.format(unique_token, class_token),
            'a shiny {0} {1}'.format(unique_token, class_token),
            'a wet {0} {1}'.format(unique_token, class_token),
            'a cube shaped {0} {1}'.format(unique_token, class_token)
            ]
        elif key in live_subjects_map:
            class_token=objects_map[key]
            self.prompt_list = [
                'a {0} {1} in the jungle'.format(unique_token, class_token),
                'a {0} {1} in the snow'.format(unique_token, class_token),
                'a {0} {1} on the beach'.format(unique_token, class_token),
                'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
                'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
                'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
                'a {0} {1} with a city in the background'.format(unique_token, class_token),
                'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
                'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
                'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
                'a {0} {1} wearing a red hat'.format(unique_token, class_token),
                'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
                'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
                'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
                'a {0} {1} in a chef outfit'.format(unique_token, class_token),
                'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
                'a {0} {1} in a police outfit'.format(unique_token, class_token),
                'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
                'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
                'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
                'a red {0} {1}'.format(unique_token, class_token),
                'a purple {0} {1}'.format(unique_token, class_token),
                'a shiny {0} {1}'.format(unique_token, class_token),
                'a wet {0} {1}'.format(unique_token, class_token),
                'a cube shaped {0} {1}'.format(unique_token, class_token)
                ]
            
        self.image_list=[]
        for n in range(5):
            try:
                img=load_image(f"https://raw.githubusercontent.com/google/dreambooth/refs/heads/main/dataset/{key}/0{n}.jpg")
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
    lay_dict={k:v[0] for k,v in get_feature_dict("stablediffusionapi/realistic-vision-v51",device).items()}
        
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
    
    
    unique_token="<sks>"
    class_token="dog"
    
    prompt_list = [
    'a {0} {1} in the jungle'.format(unique_token, class_token),
    'a {0} {1} in the snow'.format(unique_token, class_token),
    'a {0} {1} on the beach'.format(unique_token, class_token),
    'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
    'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
    'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
    'a {0} {1} with a city in the background'.format(unique_token, class_token),
    'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
    'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
    'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
    'a {0} {1} wearing a red hat'.format(unique_token, class_token),
    'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
    'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
    'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
    'a {0} {1} in a chef outfit'.format(unique_token, class_token),
    'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
    'a {0} {1} in a police outfit'.format(unique_token, class_token),
    'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
    'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
    'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
    'a red {0} {1}'.format(unique_token, class_token),
    'a purple {0} {1}'.format(unique_token, class_token),
    'a shiny {0} {1}'.format(unique_token, class_token),
    'a wet {0} {1}'.format(unique_token, class_token),
    'a cube shaped {0} {1}'.format(unique_token, class_token)
    ]
    
    data=[]
    
    start_epoch=1
    for epoch in range(start_epoch, args.epochs+1):
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
                    model_pred = hooked_unet(
                        sae_src_list,
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