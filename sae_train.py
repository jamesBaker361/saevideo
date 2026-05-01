import torch
from experiment_helpers.gpu_details import print_details
from experiment_helpers.init_helpers import repo_api_init,default_parser,parse_args
from experiment_helpers.loop_decorator import optimization_loop
import time
import os
from experiment_helpers.data_helpers import split_data
from torch.utils.data import Dataset, DataLoader,random_split
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.argprint import print_args
from datasets import load_dataset
from overcomplete.sae import TopKSAE,QSAE, JumpSAE, BatchTopKSAE,losses,SAE
from overcomplete.sae.trackers import DeadCodeTracker
import numpy as np
import torch.nn.functional as F
from unet_autopsy import get_shape_dict
from collections import defaultdict
import json
from typing import Optional
from torchvision.transforms import Resize

#https://github.com/KempnerInstitute/overcomplete

parser=default_parser({
    "project_name":"sae",
    "src_dataset":"jbaker361/filler",
    "repo_id":"jlbaker361/sae-test",
})

KSAE="ksae"
JUMP="jump"
BATCHK="batch_k"
QUANTIZED="quantized"

parser.add_argument("--nb_concepts",type=int,default=10000,help="n concepts for SAE")
parser.add_argument("--sae_model",type=str,default=KSAE)
parser.add_argument("--model_layer",type=str,default="up_blocks.1.attentions.0")
parser.add_argument("--src_dir_list",nargs="*",default=["seg_ip_jlbaker361_mtg"])
parser.add_argument("--use_dino",action="store_true",help="whether to use dino embeddings too")
parser.add_argument("--use_mask",action="store_true",help="whether to mask out irrelevant tensors")
parser.add_argument("--flatten",action="store_true",help="whether to do the whole image at once or do channel by channel")
parser.add_argument("--timestep",type=int,default=2,help="which timestep to use (probably doesn't matter)")
parser.add_argument("--hf_data",action="store_true")
parser.add_argument("--dino_coefficient",type=float,default=0.1)
parser.add_argument("--pooling",type=str,default="avg")
parser.add_argument("--threshold",type=float,default=0.5)
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7",help=" dreamshaper ")
parser.add_argument("--step",type=int,default=0)
parser.add_argument("--prefix",type=str,default="seg_ip_test")

import torch

def feature_similarity(W_old, W_new):
    # normalize features
    W_old = torch.nn.functional.normalize(W_old, dim=1)
    W_new = torch.nn.functional.normalize(W_new, dim=1)

    # similarity matrix: (old_features x new_features)
    return W_old @ W_new.T

from scipy.optimize import linear_sum_assignment

def match_features(sim_matrix):
    sim = sim_matrix.detach().cpu().numpy()
    cost = -sim  # maximize similarity → minimize negative
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind

def track_features(W_old, W_new):
    sim = feature_similarity(W_old, W_new)
    old_idx, new_idx = match_features(sim)

    matched_sim = sim[old_idx, new_idx]

    return old_idx, new_idx, matched_sim

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self,hf_dataset,model_layer):
        super().__init__()
        self.data=load_dataset(hf_dataset)
        self.model_layer=model_layer
        
    def __len__(self):
        return len(self.data[self.model_layer])
    
    def __getitem__(self, index):
        return torch.tensor(self.data[self.model_layer][index])
    
import re
import os

def get_num(path):
    name = os.path.basename(path)
    nums = re.findall(r'\d+', name)
    return int(nums[-1])  # last number in filename
    
class LatentLocalDataset(torch.utils.data.Dataset):
    def __init__(self,src_dir_list:Optional[str|list[str]],
                 model_layer:str,
                 use_mask:bool):
        
        super().__init__()
        if type(src_dir_list)==str:
            src_dir_list=[src_dir_list]
            
        self.model_layer=model_layer
        self.src_dir_list=src_dir_list
        self.use_mask=use_mask

        self.np_list = []
        for src_dir in src_dir_list:
            self.np_list+=sorted([
                os.path.join(src_dir,f) for f in os.listdir(src_dir) if f.endswith("npz")
            ],key=get_num)
        

    def __len__(self):
        return len(self.np_list)
    
    def __getitem__(self, index):
        npz_dict=np.load(self.np_list[index])
        inputs=npz_dict["input."+self.model_layer]
        print("inputs shape",inputs.shape)
        outputs=npz_dict["output."+self.model_layer]
        print("outputs shape",outputs.shape)
        diff=outputs-inputs
        try:
            mask=npz_dict["mask."+self.model_layer+".transformer_blocks.0.attn2.processor"]
            print("mask shape ",mask.shape)
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            mask[mask < 0.5] = 0.
            mask[mask>1e-4]=1.
        except KeyError as e:
            print(e)
            print(self.np_list[index])
            raise e
        
        valid = (mask != 0)
        print("valid shape ",valid.shape)
        diff = diff[:, :, valid]
        print("diff[:, :, valid] shape ",diff.shape)
        
        diff=torch.tensor(diff).permute(0,2,1).flatten(0,1) #b,c,h,w -> bhw,c
        dino=torch.tensor(npz_dict["dino"])
        dino=dino[:,0,:].expand(diff.size()[0],384)
        
        return {
            "act":diff,
            "dino":dino
        }
        
    

def main(args):
    api,accelerator,device=repo_api_init(args)
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
    src_dataset : str = args.src_dataset
    load_hf  = args.load_hf
    nb_concepts : int = args.nb_concepts
    sae_model : str = args.sae_model
    model_layer : str = args.model_layer
    src_dir_list  = args.src_dir_list
    use_dino  = args.use_dino
    use_mask  = args.use_mask
    flatten  = args.flatten
    timestep : int = args.timestep
    hf_data  = args.hf_data
    dino_coefficient : float = args.dino_coefficient
    pooling : str = args.pooling
    threshold : float = args.threshold
    checkpoint : str = args.checkpoint
    step : int = args.step
    prefix : str = args.prefix
    
    sae_model_class={
        KSAE:TopKSAE,
        JUMP:JumpSAE,
        BATCHK:BatchTopKSAE,
        QUANTIZED:QSAE
    }[args.sae_model]
    
    criterion={
        KSAE:losses.mse_l1,
        JUMP:losses.mse_l1,
        BATCHK:losses.mse_l1,
        QUANTIZED:losses.mse_l1,
    }.get(args.sae_model)
    if criterion is None:
        raise NotImplementedError(f"No loss defined for sae_model={args.sae_model}")
    
    istopk=(sae_model==KSAE)

    if len(src_dir_list) ==0:
        src_dir_list=["seg_ip_flickr","seg_ip_huggan_wikiart"]
    shape_dict=get_shape_dict(args.checkpoint,device)
    dataset= LatentLocalDataset(src_dir_list,
                                args.model_layer,
                                use_mask)
    
    train_loader,test_loader,val_loader=random_split(dataset,[0.9,0.05,0.05])
    
    for batch in train_loader:
        break
    
    print("real activation size ",batch["act"].size())
    
    
    print(shape_dict[args.model_layer])
    (b,c,h,w)=shape_dict[args.model_layer]
    
    sae_model:SAE=sae_model_class(c,args.nb_concepts,device=device)
        
    
    params=[p for p in sae_model.parameters()]
    model_dict={
       "sae" :sae_model
    }
    
    if args.use_dino:
        dino=batch["dino"]
        print("dino size ",dino.size())
        (b,dc)=dino.size() #(b,384hw)
        dino_sae_model:SAE=sae_model_class(dc,args.nb_concepts,device=device)
        params.extend([p for p in dino_sae_model.parameters()])
        model_dict["dino_sae"]=dino_sae_model
        
    act=batch["act"]
    with accelerator.autocast():
        z_pre, z, x_hat=sae_model(act.to(device))
    dead_tracker = DeadCodeTracker(z.size()[1], device)
    
    for t,name in zip([z_pre, z, x_hat],['z_pre', 'z', 'x_hat']):
        print(name,t.size())

    optimizer=torch.optim.AdamW(params,args.lr)
    
    if args.use_dino:
        dino_sae_model,optimizer,sae_model,train_loader,test_loader,val_loader = accelerator.prepare(dino_sae_model,optimizer,sae_model,train_loader,test_loader,val_loader)
    else:
        optimizer,sae_model,train_loader,test_loader,val_loader = accelerator.prepare(optimizer,sae_model,train_loader,test_loader,val_loader)

    save_subdir=os.path.join("sae_model")
    os.makedirs(save_subdir,exist_ok=True)
    save_subdir=os.path.join("sae_model",args.prefix)
    os.makedirs(save_subdir,exist_ok=True)
    save_subdir=os.path.join("sae_model",args.prefix,args.model_layer)
    os.makedirs(save_subdir,exist_ok=True)
    


    def save(epoch: int):
        os.makedirs(save_subdir, exist_ok=True)

        weights_path = os.path.join(save_subdir, "weights.pt")
        config_path = os.path.join(save_subdir, "config.json")

        # save model weights
        torch.save(sae_model.state_dict(), weights_path)

        # optional DINO weights
        if args.use_dino:
            dino_weights_path = os.path.join(save_subdir, "dino_weights.pt")
            torch.save(dino_sae_model.state_dict(), dino_weights_path)

        # save config
        with open(config_path, "w") as file:
            json.dump({
                "epoch": epoch
            }, file)


    def load(*_args,**_kwargs):
        load_dir=save_subdir
        weights_path = os.path.join(load_dir, "weights.pt")
        config_path = os.path.join(load_dir, "config.json")
        try:
            # load weights
            sae_model.load_state_dict(torch.load(weights_path))

            # optional DINO
            if args.use_dino:
                dino_weights_path = os.path.join(load_dir, "dino_weights.pt")
                if os.path.exists(dino_weights_path):
                    dino_sae_model.load_state_dict(torch.load(dino_weights_path))

            # load config
            with open(config_path, "r") as file:
                config = json.load(file)

            return config.get("epoch", 1)
        except:
            return 1
        
    
    start_epoch=load(False)
    
    old_weights=None

    @optimization_loop(
        accelerator,train_loader,args.epochs,args.val_interval,args.limit,
        val_loader,test_loader,save,start_epoch
    )
    def batch_function(batch,training,helpful_dict):
        nonlocal old_weights
        
        if helpful_dict["b"] == 0:
            new_weights = sae_model.get_dictionary().detach()

            if old_weights is not None:
                old_idx, new_idx, matched_sim = track_features(old_weights, new_weights)

                print(
                    "epoch:",
                    helpful_dict["epochs"],
                    "stability:",
                    matched_sim.mean().item()
                )
                accelerator.log({"sim":matched_sim.mean().item()})

            old_weights = new_weights.clone()
            
            if istopk:
                pass #reduce k
            
        
        activations=batch["act"].to(device)
        #activations=activations.flatten(0,1)
            
        if args.use_dino:
            dino=batch["dino"].to(device)
            dino=dino.flatten(0,1)
        if training:
            
            models=[sae_model]
            if args.use_dino:
                models.append(dino_sae_model)
            with accelerator.accumulate(*models):
                with accelerator.autocast():
                    optimizer.zero_grad()
                    z_pre, z, x_hat=sae_model(activations)
                    
                    bhw=z.size()[0]
                    
                    loss=criterion(activations, x_hat, z_pre, z, sae_model.get_dictionary())
                    
                    logging_dict={
                        "loss":loss.cpu().detach().numpy()
                    }
                    
                    if args.use_dino: 
                        z_pre_dino, z_dino, x_hat_dino=dino_sae_model(dino)
                        
                        dino_loss=criterion(dino,x_hat_dino,z_pre_dino,z_dino,dino_sae_model.get_dictionary())
                        
                        #z_dino=z_dino.expand(bhw,-1)
                        difference=F.mse_loss(z_dino,z)
                        logging_dict["dino_loss"]=dino_loss.cpu().detach().numpy()
                        logging_dict["differences"]=difference.cpu().detach().numpy()
                        loss+=args.dino_coefficient*difference+dino_loss
                        
                    accelerator.log(logging_dict)
                    accelerator.backward(loss)
                    optimizer.step()
                    dead_tracker.update(z)
        else:
            with torch.no_grad():     
                z_pre, z, x_hat=sae_model(activations)
                    
                loss=criterion(activations, x_hat, z_pre, z, sae_model.get_dictionary())
                
                if args.use_dino: 
                    z_pre_dino, z_dino, x_hat_dino=dino_sae_model(dino)
                    
                    dino_loss=criterion(dino,x_hat_dino,z_pre_dino,z_dino,dino_sae_model.get_dictionary())
                    
                    difference=args.dino_coefficient*F.mse_loss(z_dino,z)
                    
                    loss+=difference+dino_loss
                    
                    accelerator.log({
                            "dino_loss":dino_loss.cpu().detach().numpy()
                        })
                
        return loss.cpu().detach().numpy()
        
        
    batch_function()
    
    #testing !
    # try it with *just* the sae trained here
    
    
    
if __name__=='__main__':
    print_details()
    start=time.time()
    args=parse_args(parser)
    print_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")