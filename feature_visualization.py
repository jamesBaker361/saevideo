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
device=accelerator.device

pipe =DiffusionPipeline.from_pretrained(args.checkpoint).to(accelerator.device)

unet=pipe.unet
vae=pipe.vae

for model in [unet,vae]:
    model.requires_grad_(False)

noise=torch.randn((1,3,256,256),device=device)
noise*=vae.config.scaling_factor

optimizer=AdamW([noise],lr=0.0001)

target_layer='up_blocks.3.attentions.0.proj_in'

prompt_embeds, negative_prompt_embeds=pipe.encode_prompt(" ",device,1,False)
layer=100


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
        
def tv_loss(x):
    return (
        (x[:,:,:-1,:] - x[:,:,1:,:]).abs().mean() +
        (x[:,:,:,:-1] - x[:,:,:,1:]).abs().mean()
    )
        
latent = torch.nn.Parameter(torch.randn(1,4,64,64,device=device))

optimizer = AdamW([latent], lr=0.05)

for step in range(200):

    optimizer.zero_grad()
    activations.clear()

    unet(latent, t, prompt_embeds)

    act = activations[-1]

    feature_loss = -act[:,100].mean()

    reg_tv = tv_loss(latent)
    reg_l2 = latent.pow(2).mean()

    loss = feature_loss + 0.01*reg_tv + 0.001*reg_l2

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        latent.clamp_(-4,4)
        
print('done!')

image_processor=pipe.image_processor
img=vae.decode(noise)
img=image_processor.postprocess(img)[0]
img.save("activation.png")