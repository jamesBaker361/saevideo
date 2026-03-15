from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from hook_wrapper import HookWrapper
import argparse
import accelerate
from torch.optim import AdamW
import torch

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7")

args=parser.parse_args()

accelerator =accelerate.Accelerator()


pipe =DiffusionPipeline.from_pretrained(args.checkpoint).to(accelerator.device)

unet=pipe.unet
vae=pipe.vae

for model in [unet,vae]:
    model.requires_grad_(False)

noise=torch.randn((1,3,256,256))
noise*=vae.config.scaling_factor

optimizer=AdamW(noise,lr=0.0001)

target_layer='up_blocks.3.attentions.0.proj_in'

prompt_embeds, negative_prompt_embeds=pipe.encode_prompt(" ",1,False)

timesteps,num_inference_steps=retrieve_timesteps(pipe.scheduler,10)
t=timesteps[0]

activations=[]

def save_hook(name):
    def hook(module, input, output):
        if type(output)==tuple:
            output=output[0]
        try:
            activations.append(output.detach().cpu())
        except:
            print(type(module),type(output))
    return hook

for name, module in unet.named_modules():
    if name ==target_layer:
        module.register_forward_hook(save_hook(name))
        
for steps in range(2):
    latent=vae.encode(noise).latent_dist.sample()
    unet(noise,t,prompt_embeds)