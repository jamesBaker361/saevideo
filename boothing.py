#https://github.com/surkovv/sdxl-unbox

from diffusers import DiffusionPipeline
from hook_wrapper import HookForward
from overcomplete import TopKSAE
import torch
import numpy as np
import os
from accelerate import Accelerator



acc= Accelerator()
device=acc.device


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
lay_dict={}
for lay in layers:
    try:
        raw_activations=torch.tensor(np.load(os.path.join(src_dir, "0","0.npz"))[lay][0])
    except:
        for key in np.load(os.path.join(src_dir, "0","act_0.npz")).keys():
            raw_activations=torch.tensor(np.load(os.path.join(src_dir, "0","act_0.npz"))[lay][0])
    act_size=raw_activations.size()
    (c,h,w)=act_size
    c_list.append(c)
    print(lay,act_size)
    lay_dict[lay]=c
    
def get_unet_device_dtype(unet):
    param = next(unet.parameters())
    return param.device, param.dtype

device, dtype = get_unet_device_dtype(pipe.unet)
sae_dict={
    lay: TopKSAE(c,nb_concepts,).to(device,dtype) for lay,c in lay_dict.items()
}



for lay,ksae in sae_dict.items():
    try:
        ksae.load_state_dict(torch.load(
            os.path.join("sae_models",f"{src_dir}_{lay}_{step}","weights.pt")
            ))
    except:
        pass



hooker=HookForward(pipe,layers,sae_dict,lay_dict,0.5)

batch_size=1
sae_src_list={ lay: torch.normal(0,1,(batch_size,nb_concepts)).to(device,dtype)  for lay in layers}

hooker.forward(sae_src_list,"walking",height=64,width=64,num_inference_steps=4)