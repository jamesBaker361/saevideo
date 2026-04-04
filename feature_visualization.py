from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import argparse
import accelerate
from torch.optim import AdamW
import torch
import time
from experiment_helpers.gpu_details import print_details
from overcomplete.sae import TopKSAE,QSAE, JumpSAE, BatchTopKSAE,losses,SAE
from overcomplete.sae.trackers import DeadCodeTracker
import wandb
import os
from unet_autopsy import get_shape_dict

KSAE="ksae"
JUMP="jump"
BATCHK="batch_k"
QUANTIZED="quantized"


parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7")
parser.add_argument("--epochs",type=int,default=10000)
parser.add_argument("--target_layer",type=str,default='up_blocks.1.attentions.0')
parser.add_argument("--sae_checkpoint",type=str,default="sae_model/seg_ip_flickr_up_blocks.1.attentions.0_2/weights.pt")
parser.add_argument("--lr",type=float,default=0.05)
parser.add_argument("--num_inference_steps",type=int,default=4)
parser.add_argument("--channel",type=int,default=0,help="channel to optimize")
parser.add_argument("--size",type=int,default=64,help="latent size (actual image will be size * 8)")
parser.add_argument("--save_dir",type=str,default="features")
parser.add_argument("--sae_model",type=str,default=KSAE)
parser.add_argument("--nb_concepts",type=int,default=10000,help="n concepts for SAE")
parser.add_argument("--project_name",type=str,default="feature")



def main(args):
    accelerator =accelerate.Accelerator(log_with="wandb")
    accelerator.init_trackers(args.project_name)
    device=accelerator.device
    
    if args.sae_checkpoint.find(args.target_layer)==-1:
        print("args.sae_checkpoint.find(args.target_layer) ==-1 this looks might be an error ")

    pipe =DiffusionPipeline.from_pretrained(args.checkpoint).to(accelerator.device)

    unet=pipe.unet
    vae=pipe.vae

    for model in [unet,vae]:
        model.requires_grad_(False)

    target_layer=args.target_layer

    prompt_embeds, negative_prompt_embeds=pipe.encode_prompt(" ",device,1,False)

    sae_model_class={
        KSAE:TopKSAE,
        JUMP:JumpSAE,
        BATCHK:BatchTopKSAE,
        QUANTIZED:QSAE
    }[args.sae_model]

    timesteps,num_inference_steps=retrieve_timesteps(pipe.scheduler,args.num_inference_steps)

    activations=None

    def save_hook(name):
        def hook(module, input, output):
            nonlocal activations
            if type(output)==tuple:
                output=output[0]
            setattr(module,"cached_output",output)
        return hook

    for name, module in unet.named_modules():
        if name ==target_layer:
            module.register_forward_hook(save_hook(name))
            break
            
    print("module ",module)
            
    def tv_loss(x): #total variation loss
        return (
            (x[:,:,:-1,:] - x[:,:,1:,:]).abs().mean() +
            (x[:,:,:,:-1] - x[:,:,:,1:]).abs().mean()
        )
            
    latent = torch.nn.Parameter(vae.config.scaling_factor* torch.randn(1,4,args.size,args.size,device=device))
    
    output_dict=get_shape_dict(args.checkpoint,device)
    
    (b,c,h,w)=output_dict[target_layer]
    
    sae:SAE=sae_model_class(c,args.nb_concepts)
    sae.load_state_dict(torch.load(args.sae_checkpoint,map_location=device))
    sae.to(device)
    sae.eval()
    
    os.makedirs(args.save_dir,exist_ok=True)
    start_channel=len([f for f in os.listdir(args.save_dir) if f.endswith("png")])
    
    for channel in range(start_channel,args.nb_concepts):
        latent = torch.nn.Parameter(vae.config.scaling_factor* torch.randn(1,4,args.size,args.size,device=device))
    
        optimizer = AdamW([latent], lr=args.lr)

        for step in range(args.epochs):
            for t in timesteps:

                optimizer.zero_grad()

                unet(latent, t, prompt_embeds)

                act = getattr(module,"cached_output")
                act:torch.Tensor=act.permute((0,2,3,1))
                #print("act size",act.size())
                act=act.flatten(0,2)
                act=sae.encode(act)[0]
                #print("act size",act.size())

                feature_loss = -act[:,channel].mean()

                reg_tv = tv_loss(latent)
                reg_l2 = latent.pow(2).mean()

                loss = feature_loss + 0.01*reg_tv + 0.001*reg_l2

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    latent.clamp_(-4,4)
                    
            if step%100==0:
                image_processor=pipe.image_processor
                img=vae.decode(latent).sample.cpu().detach()
                img=image_processor.postprocess(img)[0]
                accelerator.log({
                    f"img_{channel}":wandb.Image(img)
                })
                
        print(f"finished image {channel}")
        with torch.no_grad():
            image_processor=pipe.image_processor
            img=vae.decode(latent).sample.cpu().detach()
            img=image_processor.postprocess(img)[0]
            accelerator.log({
                    f"img_{channel}":wandb.Image(img)
            })
            save_path=os.path.join(args.save_dir,f"{channel}.png")
            img.save(save_path)
    
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