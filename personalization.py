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
from diffusers import UNet2DConditionModel
from sdxl_pipe import HookedStableDiffusionXLWithUNetPipeline

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
from diffusers.image_processor import VaeImageProcessor
import wandb

parser=default_parser()
parser.add_argument("--layers",nargs="*",default=["down_blocks.1.attentions.1","down_blocks.2.attentions.1"])
parser.add_argument("--hidden_dim",nargs='*',help=" hidden dim of sae, if len = 1 then we default to all of them being the one thing")
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7")
parser.add_argument("--nb_concepts",type=int,default=10000,help="n concepts for SAE")
parser.add_argument("--parent_dir",type=str,default="sae_model")
parser.add_argument("--prefix",type=str,default="seg_ip_txt_")
parser.add_argument("--use_mask",action="store_true",help="use use_mask to generate images")
parser.add_argument("--step",type=int,default=2)
parser.add_argument("--subset",type=str,default="subject",help="subject or object or face")
parser.add_argument("--size",type=int,default=256)
parser.add_argument("--num_inference_steps",type=int,default=8)
parser.add_argument("--weight",type=float,default=0.95)
parser.add_argument("--use_dino",action="store_true")
parser.add_argument("--add_activation",action="store_true")
parser.add_argument("--use_mean",action="store_true")
#TODO: add use_mask patching
            
#use persona dataset


        
        

def main(args):
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
    layers  = args.layers
    hidden_dim  = args.hidden_dim
    checkpoint : str = args.checkpoint
    nb_concepts : int = args.nb_concepts
    parent_dir : str = args.parent_dir
    prefix : str = args.prefix
    use_mask  = args.use_mask
    step : int = args.step
    subset : str = args.subset
    size : int = args.size
    num_inference_steps : int = args.num_inference_steps
    weight:float=args.weight
    use_dino:bool=args.use_dino
    add_activation:bool=args.add_activation
    use_mean:bool=args.use_mean
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
    
    if use_dino:
    
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
    
    if torch.cuda.is_available():

        pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
            'stabilityai/sdxl-turbo',
            #torch_dtype=dtype,
            device_map="balanced",
            #variant=("fp16" if dtype==torch.float16 else None)
        )
    else:
         pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
            'stabilityai/sdxl-turbo',
            #torch_dtype=dtype,
            device_map="cpu",
            #variant=("fp16" if dtype==torch.float16 else None)
        )
    #TODO: do this shit but just use hooks like a normal person
    
    vae=pipe.vae
    text_encoder=pipe.text_encoder
    unet=pipe.unet
    scheduler=pipe.scheduler
    image_processor=VaeImageProcessor()
    
    block_dict={}
    CACHED_ACTIVATIONS="cached_activations"
    CACHED_OUTPUTS="cached_outputs"
    unet: UNet2DConditionModel =pipe.unet
    mask_threshold=0.5
    n_tokens=2
    for layer,mod in unet.named_modules():
        if layer in layers:

            def hook_fn(module,input,output):
                activations=getattr(module,CACHED_ACTIVATIONS,None)
                if activations is None:
                    if type(output)==tuple:
                        act=output[0]
                    else:
                        act=output
                    setattr(module,CACHED_ACTIVATIONS,act)
                    return output
                if type(output)==tuple:
                    dims=output[0].size()[-2:]
                else:
                    dims=output.size()[-2:]
                    
                mask=torch.ones(dims).unsqueeze(0).unsqueeze(0)
                
                if use_mask:
                    key=getattr(module.attn2.to_k,CACHED_OUTPUTS)
                    query=getattr(module.attn2.to_q,CACHED_OUTPUTS)
                    attn_heads=module.attn2.heads
                    
                    inner_dim = key.shape[-1]
                    head_dim = inner_dim // attn_heads

                    query = query.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
                    key = key.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
                    
                    attn_weight = query @ key.transpose(-2, -1)
                    attn_weight = torch.softmax(attn_weight, dim=-1)


                    mask=attn_weight.mean(dim=1).view(batch_size, dims,-1)[:,:,:,1:1+n_tokens].mean(dim=-1) #shape B, h, w
                    
                    mask_min=mask.min()
                    mask_max=mask.max()
                    mask =(mask-mask_min)/(mask_max-mask_min+1e-6)
                    
                    mask[mask<mask_threshold]=0.
                    mask=mask.unsqueeze(1)
                
                mask*=weight
                

                if type(output)==tuple:
                    out=(1-mask)*output[0] + mask*activations
                    if len(output)==1:
                        return (out,)
                    else:
                        return (out, * output[1:])
                else:
                    return (1-mask)*output + mask*activations
                
            mod.register_forward_hook(hook_fn)
            block_dict[layer]=mod
            if use_dino:
                pass #gotta do some training and shit for this
                
                
                
        elif layer.find("attn2.to")!=-1:
            def hook_kqv(module,input,output):
                setattr(module,CACHED_OUTPUTS,output)
                return output
            
            mod.register_forward_hook(hook_kqv)
                    
    
    
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
            
            if use_dino:
                dino=get_last_hidden_states(image,dino_processor,dino_model)[:, 0, :].to(device)
                sae_src_dict={
                    layer: sae_dict[layer].decode(dino_sae_dict[layer].encode(dino)[1]) for layer in dino_sae_dict
                }
            else:
                for mod in block_dict.values():
                    setattr(mod,CACHED_ACTIVATIONS,None) #reset the cached activations for each src image
                image_pt=image_processor.preprocess(image,size,size)
                latents=vae.config.scaling_factor*vae.encode(image_pt).latent_dist.sample()
                if torch.isnan(latents).any():
                    print("is nan latents ")
                noise = torch.randn_like(latents)
                
                timesteps = torch.randint(
                    0, 2, (latents.shape[0],), device=latents.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = scheduler.add_noise(latents, noise, timesteps)
                
                (prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                )=pipe.encode_prompt(prompt,prompt,device,1,False," "," ")
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
                        
                unet.forward(
                    noisy_model_input,timesteps,
                                        encoder_hidden_states=prompt_embeds,
                                        timestep_cond=timestep_cond,
                                        added_cond_kwargs=added_cond_kwargs,
                                        return_dict=False,
                )[0]
                
            result=pipe.forward(prompt,num_inference_steps=num_inference_steps,height=size,width=size,return_dict=True,output_type="pil").images[0]
            
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