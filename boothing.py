#https://github.com/surkovv/sdxl-unbox
#https://huggingface.co/surokpro2/sdxl-saes/tree/main


import torch
import numpy as np
from experiment_helpers.argprint import print_args
import json
from PIL import Image
from loading import get_sae_dict
from sdxl_unbox.SAE import SparseAutoencoder
from sdxl_pipe import HookedStableDiffusionXLWithUNetPipeline
from diffusers import UNet2DConditionModel
from diffusers import DiffusionPipeline, AutoencoderKL
import wandb


import os
import argparse
from experiment_helpers.gpu_details import print_details
import sys
sys.stdout.flush()

import time
import torch.nn.functional as F

from experiment_helpers.init_helpers import default_parser,repo_api_init

from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import DataLoader,Dataset
from unet_autopsy import get_shape_dict

unique_token="<sks>"

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
    def __init__(self,key:str,text_encoder,tokenizer,size:int):
        super().__init__()
        self.image_processor=VaeImageProcessor()
        self.tokenizer=tokenizer
        self.text_encoder=text_encoder

        
        
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
                self.image_list.append(img.resize((size,size)))
            except:
                break
            
    def __len__(self):
        return len(self.prompt_list)
    
    def __getitem__(self, index):
        return {
            "prompt":self.prompt_list[index],
            "image":self.image_processor.preprocess(self.image_list[index %len(self.image_list)]),
            "input_ids":compute_text_embeddings(self.prompt_list[index],self.tokenizer,self.text_encoder)
        }
        
                  
        
        
        

parser=default_parser({"epochs":3})

parser.add_argument("--weight",type=float,default=0.01)
parser.add_argument("--key",type=str,default="chair")
parser.add_argument("--checkpoint",type=str,default="stablediffusionapi/realistic-vision-v51")
parser.add_argument("--num_inference_steps",type=int,default=2)
parser.add_argument("--nb_concepts",type=int,default=10000,help="n concepts for SAE")
parser.add_argument("--prefix",type=str,default="features_stablediffusionapi_realistic-vision-v51_32_")
parser.add_argument("--step",type=int,default=24)
parser.add_argument("--size",type=int,default=64)
parser.add_argument("--mask_threshold",type=float,default=0.5)
parser.add_argument("--use_attn_mask",action="store_true")
parser.add_argument("--n_tokens",type=int,default=2)
parser.add_argument("--use_mean",action="store_true")
parser.add_argument("--use_bias",action="store_true")



def main(args):
    api,accelerator,device=repo_api_init(args)
    
    mixed_precision : str = args.mixed_precision
    project_name : str = args.project_name
    gradient_accumulation_steps : int = args.gradient_accumulation_steps
    repo_id : str = args.repo_id
    lr : float = args.lr
    epochs : int = args.epochs
    limit : int = args.limit
    save_dir : str = args.save_dir
    batch_size : int = args.batch_size
    val_interval : int = args.val_interval
    load_hf  = args.load_hf
    weight : float = args.weight
    key : str = args.key
    checkpoint : str = args.checkpoint
    num_inference_steps : int = args.num_inference_steps
    nb_concepts : int = args.nb_concepts
    prefix : str = args.prefix
    step : int = args.step
    size:int=args.size
    use_attn_mask:bool=args.use_attn_mask
    mask_threshold:float=args.mask_threshold
    n_tokens:int=args.n_tokens
    use_mean:bool=args.use_mean
    use_bias:bool=args.use_bias
    os.makedirs(args.save_dir,exist_ok=True)



    dtype=torch.float16

    if torch.cuda.is_available():

        pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
            'stabilityai/sdxl-turbo',
            torch_dtype=dtype,
            device_map="balanced",
            variant=("fp16" if dtype==torch.float16 else None)
        )
    else:
         pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
            'stabilityai/sdxl-turbo',
            torch_dtype=dtype,
            device_map="cpu",
            variant=("fp16" if dtype==torch.float16 else None)
        )

    path_to_checkpoints = './sdxl_unbox/checkpoints/'

    

    block_list=[
        "down_blocks.2.attentions.1",
        "mid_block.attentions.0",
        "up_blocks.0.attentions.0",
         "up_blocks.0.attentions.1"
    ]
    
    saes_dict:dict[str,SparseAutoencoder] = {}
    means_dict = {}

    shape_dict=get_shape_dict('stabilityai/sdxl-turbo',device,size)
    trainable_embedding_dict={}
    for block in block_list:
        try:
            sae = SparseAutoencoder.load_from_disk(
                os.path.join(path_to_checkpoints, f"unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final"),
            )
            if torch.isnan(sae.decoder.weight).any():
                print("nan decoder weight ",block)
        except RuntimeError:
            sae = SparseAutoencoder.load_from_disk(
                os.path.join(path_to_checkpoints, f"unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final"),map_location=torch.device('cpu')
            )
        means = torch.load(
            os.path.join(path_to_checkpoints, f"unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final", "mean.pt"),
            weights_only=True
        )
        saes_dict[block] = sae.to(device, dtype=dtype)
        saes_dict[block].requires_grad_(False)
        print(block,shape_dict[block])
        print(means.size(), means.max(),means.min(), means.std(),means.mean())
        trainable_embedding_dict[block]=means.to(device, dtype=dtype)
        means.requires_grad_(True)
        
    def get_unet_device_dtype(unet):
        param = next(unet.parameters())
        return param.device, param.dtype

    device, dtype = get_unet_device_dtype(pipe.unet)

    pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device)

    vae=pipe.vae
    text_encoder=pipe.text_encoder
    unet=pipe.unet
    scheduler=pipe.scheduler
    tokenizer=pipe.tokenizer
    
    for model in [vae,text_encoder,unet]:
        model.requires_grad_(False)
        
    unet: UNet2DConditionModel =pipe.unet
    
    for block in block_list:
        trainable_embedding=trainable_embedding_dict[block]
        sae=saes_dict[block]
        
        recons=trainable_embedding @ sae.decoder.weight.T
        
        if torch.isnan(recons).any():
            print(block, " initial recons thing is nan or something ")
        else:
            print("max min ",recons.max(),recons.min())
    
    unet_modules={}
    attn_modules:dict[str,torch.nn.Module]={}
    for name,module in unet.named_modules():
        if name in block_list:
            unet_modules[name]=module
        if name.find("attn1.to")!=-1 or name.find("attn2.to")!=-1:
            attn_modules[name]=module
            
    CACHE_NAME="cache"
    for attn,attn_block in attn_modules.items():
        setattr(attn_block,"dict_name",attn)
        setattr(attn_block,CACHE_NAME,None)
        def cache_kv(module,input,output):
            setattr(module,CACHE_NAME,output)
            if output is not None:
                return output
            else:
                return input
        
        attn_block.register_forward_hook(cache_kv)
    
    for block in block_list:
        unet_mod=unet_modules[block]
        sae=saes_dict[block]
        trainable_embedding=trainable_embedding_dict[block]
        setattr(unet_mod, "sae_custom",sae)
        setattr(unet_mod, "trainable_embedding",trainable_embedding)
        attn_heads=unet_mod.transformer_blocks[0].attn2.heads
        
        def feature_injection(module, input, output, block=block, sae=sae,attn_heads=attn_heads):
            
            
            
            #print("feature injection called with ")
            trainable_embedding=trainable_embedding_dict[block]
            if torch.isnan(trainable_embedding).any():
                print("nan trainable embedding")
            sae=saes_dict[block]
            mean=means_dict[block]
            if use_mean:
                recons=trainable_embedding-mean
                recons=recons @ sae.decoder.weight.T
            else:
                recons = trainable_embedding @ sae.decoder.weight.T
                
            if torch.isnan(recons).any():
                print("recons nan")
            else:
                print("recons okay :)")
            if use_bias:
                recons=recons+sae.pre_bias
            recons=recons.unsqueeze(-1).unsqueeze(-1)
            
            if use_attn_mask:
                to_k=module.transformer_blocks[0].attn2.to_k
                key=getattr(to_k,CACHE_NAME)
                to_q=module.transformer_blocks[0].attn2.to_q
                query=getattr(to_q,CACHE_NAME)
                
                batch_size=key.size()[0]
                [h,w]=shape_dict[block][2:]
                
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn_heads

                query = query.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
                #print("\t query size",query.size())

                key = key.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
    

                #print("\t hidden states shape after scaled dot product",hidden_states.size())
                attn_weight = query @ key.transpose(-2, -1)
                attn_weight = torch.softmax(attn_weight, dim=-1)

                mask=attn_weight.mean(dim=1).view(batch_size, h,w,-1)[:,:,:,:n_tokens].mean(dim=-1) #shape B, h, w
                
                mask_min=mask.min()
                mask_max=mask.max()
                mask =(mask-mask_min)/(mask_max-mask_min+1e-6)
                
                mask[mask<mask_threshold]=0.
                mask=mask.unsqueeze(1)
            else:
                mask=torch.ones_like(recons)
            mask[mask>0]=weight
            
            if type(input)==tuple:
                for i,input_tensor in enumerate(input):
                    if torch.isnan(input_tensor).any():
                        print("nan input ",i)
                        
            if type(output)==tuple:
                for o,output_tensor in enumerate(output):
                    if torch.isnan(output_tensor).any():
                        print("nan output ",o)
                        
            #return output
            
            
            if type(output)==tuple:
                
                if type(input) ==tuple:
                    input=input[0]
                original=output[0]-input
                
                if torch.isnan(original).any():
                    print("orignal nan")
                
                if len(output)>=2:
                    
                    return ((mask * recons) + ((1-mask) * original)+input, *output[1:])
                else:
                    return ((mask * recons) + ((1-mask) * original)+input,)
            else:
                original=output-input
                return (mask * recons) + ((1-mask) * original)+input
        unet_mod.register_forward_hook(feature_injection)
        
                    
    
    
    params=[v for v in trainable_embedding_dict.values()]
    optimizer_class = torch.optim.AdamW
    optimizer=optimizer_class(params,lr)
    optimizer.zero_grad()
    
    
    data=DreamboothDataset(args.key,text_encoder,tokenizer,size)
    
    
    start_epoch=1
    for epoch in range(start_epoch, args.epochs+1):
        loss_list=[]
        for r,row in enumerate(data):
            if r==limit:
                break
            img=row["image"].to(device,dtype)
            #print("img max min ",img.max(),img.min())
            #print("img size",img.size())
            latents=vae.config.scaling_factor*vae.encode(img).latent_dist.sample()
            if torch.isnan(latents).any():
                print("is nan latents ")
            noise = torch.randn_like(latents)
            
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = scheduler.add_noise(latents, noise, timesteps)
            
            (prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            )=pipe.encode_prompt(row["prompt"],row["prompt"],device,1,False," "," ")
            timestep_cond=None
            add_text_embeds = pooled_prompt_embeds

            if pipe.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

            original_size = (size, size)
            target_size =(size, size)
            crops_coords_top_left=(0,0)
            add_time_ids = pipe._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,)

            actual_batch_size = noisy_model_input.shape[0]
            prompt_embeds = prompt_embeds.expand(actual_batch_size, -1, -1).contiguous()
            add_text_embeds = add_text_embeds.expand(actual_batch_size, -1).contiguous()
            add_time_ids = add_time_ids.expand(actual_batch_size, -1).contiguous()
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}



            with accelerator.accumulate(params):
                with accelerator.autocast(): #possibly THIS is bad???
                    
                    model_pred = unet.forward(
                        noisy_model_input,timesteps,
                                            encoder_hidden_states=prompt_embeds,
                                            timestep_cond=timestep_cond,
                                            added_cond_kwargs=added_cond_kwargs,
                                            return_dict=False,
                    )[0]
                    
                    if scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif scheduler.config.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(latents, noise, timesteps)
                        
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params,1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            loss_list.append(loss.cpu().detach().numpy())
            
        accelerator.log({
            "loss":np.mean(loss_list)
        })
        print("loss",np.mean(loss_list))
    os.makedirs(args.save_dir,exist_ok=True)
    for lay,t in trainable_embedding_dict.items():
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
    
    
    
    for p,prompt in enumerate(prompt_list):
        gen_img:Image.Image=pipe(prompt,prompt,size,size,num_inference_steps).images[0]
        gen_img.save(os.path.join(args.save_dir,f"gen_{p}.jpg"))
        accelerator.log({
            f"img_{p}":wandb.Image(gen_img)
        })
            

    #hooker.forward(sae_src_list,"walking",height=64,width=64,num_inference_steps=4)
    
if __name__=='__main__':
    print_details()
    print("current process ",os.getpid())
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