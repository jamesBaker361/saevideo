from unet_autopsy import get_shape_dict
from overcomplete import TopKSAE
import torch
import os

def get_sae_dict(checkpoint:str,device,nb_concepts:int,layers:list[str],prefix:str,step:int,weight_name:str="weights.pt"):
    shape_dict=get_shape_dict(checkpoint,device)
    
    for layer in layers:
        print(layer,shape_dict[layer])
    
    sae_dict={
        layer: TopKSAE(shape_dict[layer][0],nb_concepts) for layer in layers
    }
    
    for layer,ksae in sae_dict.items():
        if torch.cuda.is_available():
            ksae.load_state_dict(torch.load(
                    os.path.join("sae_model",f"{prefix}{layer}_{step}",weight_name)
                    ))
        else:
            ksae.load_state_dict(torch.load(
                    os.path.join("sae_model",f"{prefix}{layer}_{step}",weight_name),map_location=torch.device("cpu")
                    ))
        ksae.requires_grad_(False)
        
    return sae_dict