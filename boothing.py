#https://github.com/surkovv/sdxl-unbox

from diffusers import DiffusionPipeline
from hook_wrapper import HookForward
from overcomplete import TopKSAE
import torch
import numpy as np
import os

pipe=DiffusionPipeline.from_pretrained("stablediffusionapi/realistic-vision-v51")

layers=[
    "down_blocks.1.attentions.0","down_blocks.1.attentions.1",
                    "down_blocks.2.attentions.0","down_blocks.2.attentions.1",
                    "up_blocks.1.attentions.0","up_blocks.1.attentions.1",
                    "up_blocks.2.attentions.1","up_blocks.2.upsamplers",
                    "up_blocks.3.attentions.1","up_blocks.3.upsamplers",
                    "mid_block.attentions.0"
]
nb_concepts=10000
c_list=[]
step=24
src_dir=f"features_stablediffusionapi_realistic-vision-v51_32_{step}"
for lay in layers:
    try:
        raw_activations=torch.tensor(np.load(os.path.join(src_dir, "0","0.npz"))[lay][0])
    except:
        raw_activations=torch.tensor(np.load(os.path.join(src_dir, "0","act_0.npz"))[lay][0])
    act_size=raw_activations.size()
    (c,h,w)=act_size
    c_list.append(c)
sae_list=[
    TopKSAE(c,nb_concepts,) for c in c_list
]
for ksae in sae_list:
    try:
        ksae.load_state_dict(torch.load(
            os.path.join("sae_models",src_dir,"weights.pt")
            ))
    except:
        pass

hooker=HookForward(pipe,layers,sae_list,c_list,0.5)

batch_size=1
sae_src_list=[torch.normal(0,1,(batch_size,nb_concepts)) for _ in layers]

hooker.forward(sae_src_list,"walking",height=64,width=64,num_inference_steps=4)