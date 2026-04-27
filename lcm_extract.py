#uses ip adapter images to generate features of stuff

import os
#os.environ["TQDM_DISABLE"] = "1"
import sys
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from accelerate import Accelerator
import time

from diffusers import UNet2DConditionModel
sys.path.append(os.path.dirname(__file__))
import torch
from compatible_pipelines import CompatibleLatentConsistencyModelPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL

import datasets
from datasets import Dataset
import numpy as np

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="seg-ip-sae")
parser.add_argument("--load_hf",action="store_true",help="whether to load a special pretrained model")
parser.add_argument("--embedding",type=str, help="ignore unless load from hf; its the embedding type for embedding helpers")
parser.add_argument("--pretrained_model_path",type=str,default="")
parser.add_argument("--src_dataset",type=str, default="pixelprose/pixelprose-shards")
parser.add_argument("--subset",type=str,default="redcaps")
parser.add_argument("--use_test_split",action="store_true", help="only true for league dataset")
parser.add_argument("--initial_steps",type=int,default=4,help="how many steps for the initial inference")
parser.add_argument("--initial_mask_step_list",nargs="*",help="steps to generate mask from",type=int,default=[1,2])
parser.add_argument("--threshold",type=float,default=0.5,help="threshold for mask")
parser.add_argument("--limit",type=int,default=-1,help="limit of samples")
parser.add_argument("--layer_index",type=int,default=15)
parser.add_argument("--dim",type=int,default=256)
parser.add_argument("--token",type=int,default=1, help="which IP token is attention")
parser.add_argument("--overlap_frac",type=float,default=0.8)
parser.add_argument("--segmentation_attention_method",type=str,help="overlap or exclusive",default="overlap")
parser.add_argument("--kv_type",type=str,default="ip")
parser.add_argument("--initial_ip_adapter_scale",type=float,default=0.75)
parser.add_argument("--background",action="store_true")
parser.add_argument("--dest_dataset",type=str, default="jlbaker361/monkey-sae")
parser.add_argument("--object",type=str,default="character")
parser.add_argument("--save_dir",type=str,default="seg_ip")
parser.add_argument("--src_dir",type=str,default="synthetic_sana_synth_txt")
parser.add_argument("--hf_data",action="store_true")
parser.add_argument("--prompt_per_image",type=int,default=2)
parser.add_argument("--size",type=int,default=512)
parser.add_argument("--num_inference_steps",type=int,default=8)

#use text prompts to generate images- and use real images, and then use kv stuff to map text to locations and use that to find which text tokens 
# are commonly associated with features

def main(args):
    size:int=args.size
    num_inference_steps:int=args.num_inference_steps
    with torch.no_grad():
        prompt_txt="real_test_prompt_list.txt"
        if os.path.exists(prompt_txt):
            with open(prompt_txt,"r") as file:
                real_test_prompt_list=[s.strip() for s in file.readlines()]
        else:
            real_test_prompt_list=["in space","on a bus"]
        
        accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
        accelerator.init_trackers(project_name=args.project_name,config=vars(args))
        
        pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float16,
        ).to(accelerator.device)
        
        unet:UNet2DConditionModel =pipe.unet
        vae:AutoencoderKL=pipe.vae
        image_processor:VaeImageProcessor=pipe.image_processor
        scheduler=pipe.scheduler
        setattr(pipe,"safety_checker",None)
        
        def hook(module,input,output):
            
            if type(input)==tuple:
                for i,input_tensor in enumerate(input):
                    if torch.isnan(input_tensor).any():
                        print("nan input ",i)
                        
                setattr(module,"cached_input",input[0])
            else:
                setattr(module,"cached_input",input)
            
            if type(output)==tuple:
                for o,output_tensor in enumerate(output):
                    if torch.isnan(output_tensor).any():
                        print("nan output ",o)
                setattr(module,"cached_output",output[0])
            else:
                setattr(module,"cached_output",output)
            return output
        
        block_list=[
            "down_blocks.0.attentions.0",
            "down_blocks.1.attentions.0",
            "down_blocks.2.attentions.0",
            "down_blocks.0.attentions.1",
            "down_blocks.1.attentions.1",
            "down_blocks.2.attentions.1",
            "mid_block.attentions.0",
            "up_blocks.1.attentions.0",
            "up_blocks.2.attentions.0",
            "up_blocks.1.attentions.1",
            "up_blocks.2.attentions.1",
        ]
        block_dict={}
        for name,module in unet.named_modules():
            if name in block_list or name.find("to_k")!=-1 or name.find("to_q")!=-1 or name.find("to_v")!=-1:
                module.register_forward_hook(hook)
                block_dict[name]=module
                
        print("extracting activations from:")
        for key in block_dict:
            print('\t',key)
            
        #if args.hf_data: #Do this 
        data=datasets.load_dataset(args.src_dataset,args.subset,split="train")
        data=data.cast_column("jpg",datasets.Image())
            #were doing hf data because we need captions!!! who has captions btw: https://huggingface.co/datasets/HuggingFaceM4/NoCaps
            #data=[{"image":Image.open(os.path.join(args.src_dir,file)) }for file in path_list]
                
        os.makedirs(args.save_dir,mode=0o777,exist_ok=True)
        
        count=len([f for f in os.listdir(args.save_dir) if f.endswith("npz")])
        
        print(f"found {count} different images total data len {len(data)}")

        for k,row in enumerate(data):
            if k<count:
                continue
            if k==args.limit:
                break
            
            image=row["jpg"]
            annotations=row["json"]["vlm_caption"].lower()
            for filler in ["this photo shows","in this image",
                           "here is a","there is","this image has",
                           "this is a","this image shows","we see a",
                           "this picture shows","the image displays",]:
                annotations=annotations.replace(filler,"")
            
            npz_dict={}
            
            image_pt=image_processor.preprocess(image).to(device=accelerator.device)
            latents=vae.encode(image_pt).latent_dist.sample()
            
            for p,prompt in enumerate(annotations[:args.prompt_per_image]):
                new_image=pipe(prompt,size,size,num_inference_steps=num_inference_steps).images[0]
                for key,value in block_dict.items():
                    for attr in ["input","output"]:
                        npz_dict[f"synthetic.{p}.{key}.{attr}"]=getattr(value,f"cached_{attr}").cpu().detach().numpy()
                image_path=os.path.join(args.save_dir,f"{k}.{p}.jpg")
                new_image.save(image_path)
                
                prompt_embeds, _ = pipe.encode_prompt(
                    prompt,
                    accelerator.device,
                    1,
                    pipe.do_classifier_free_guidance,
                    negative_prompt=None,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                )
                
                timesteps = torch.randint(
                    0, 5, (latents.shape[0],), device=latents.device
                )
                timesteps = timesteps.long()
                
                noise = torch.randn_like(latents)
                noisy_model_input = scheduler.add_noise(latents, noise, timesteps)
                
                model_pred = unet.forward(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,)[0]
                
                for key,value in block_dict.items():
                    for attr in ["input","output"]:
                        npz_dict[f"{p}.{key}.{attr}"]=getattr(value,f"cached_{attr}").cpu().detach().numpy()

            np.savez(os.path.join(args.save_dir, f"{k}.npz"), **npz_dict)


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