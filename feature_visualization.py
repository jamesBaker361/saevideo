from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from hook_wrapper import HookWrapper
import argparse
import accelerate
from torch.optim import AdamW
import torch
import time
from experiment_helpers.gpu_details import print_details

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7")
parser.add_argument("--epochs",type=int,default=10000)
parser.add_argument("--target_layer",type=str,default='up_blocks.3.attentions.0.proj_in')
parser.add_argument("--lr",type=float,default=0.05)
parser.add_argument("--num_inference_steps",type=int,default=4)
parser.add_argument("--channel",type=int,default=0,help="channel to optimize")
parser.add_argument("--size",type=int,default=64,help="latent size (actual image will be size * 8)")
parser.add_argument("--save_path",type=str,default="activation.png")




def main(args):
    accelerator =accelerate.Accelerator()
    device=accelerator.device

    pipe =DiffusionPipeline.from_pretrained(args.checkpoint).to(accelerator.device)

    unet=pipe.unet
    vae=pipe.vae

    for model in [unet,vae]:
        model.requires_grad_(False)

    target_layer=args.target_layer

    prompt_embeds, negative_prompt_embeds=pipe.encode_prompt(" ",device,1,False)
    channel=args.channel


    timesteps,num_inference_steps=retrieve_timesteps(pipe.scheduler,args.num_inference_steps)
    t=timesteps[0]

    activations=[]

    def save_hook(name):
        def hook(module, input, output):
            if type(output)==tuple:
                output=output[0]
            try:
                activations=output.detach().cpu()
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
            
    latent = torch.nn.Parameter(vae.config.scaling_factor* torch.randn(1,4,args.size,args.size,device=device))

    optimizer = AdamW([latent], lr=args.lr)

    for step in range(args.epochs):

        optimizer.zero_grad()

        unet(latent, t, prompt_embeds)

        act = activations

        feature_loss = -act[:,channel].mean()

        reg_tv = tv_loss(latent)
        reg_l2 = latent.pow(2).mean()

        loss = feature_loss + 0.01*reg_tv + 0.001*reg_l2

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            latent.clamp_(-4,4)
            
    print('done!')
    with torch.no_grad():
        image_processor=pipe.image_processor
        img=vae.decode(latent).sample.cpu().detach()
        img=image_processor.postprocess(img)[0]
        img.save(args.save_path)
    
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