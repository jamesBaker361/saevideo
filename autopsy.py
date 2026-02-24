import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import accelerate


ALI=True
COSMOS=True
HUN=True
MOCHI=True
WAN=True

if ALI:
    print("#####\nali!!")

    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    #go through and dissect this one first
    print(type(pipe))
    print(dir(pipe))
    

if COSMOS:
    print("######\ncosmos!!!")
    import torch
    from diffusers import CosmosTextToWorldPipeline
    from custom_pipeline_cosmos_text2world import CustomCosmosTextToWorldPipeline
    from diffusers.utils import export_to_video

    model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
    pipe = CustomCosmosTextToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    trasnformer=pipe.transformer
    for name, module in trasnformer.named_modules():
        print(name, type(module))

if HUN:
    print("###\nHUN")
    import torch

    dtype = torch.bfloat16
    device = "cuda:0"

    from diffusers import HunyuanVideo15Pipeline
    from diffusers.utils import export_to_video

    pipe = HunyuanVideo15Pipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", torch_dtype=dtype)
    trasnformer=pipe.transformer
    for name, module in trasnformer.named_modules():
        print(name, type(module))
    
if MOCHI:
    print("###\nmochi")
    import torch
    from diffusers import MochiPipeline
    from diffusers.utils import export_to_video

    pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16)
    trasnformer=pipe.transformer
    for name, module in trasnformer.named_modules():
        print(name, type(module))
        
if WAN:
    print("###\n Wan")