import torch
from experiment_helpers.gpu_details import print_details
from experiment_helpers.init_helpers import repo_api_init,default_parser,parse_args
from experiment_helpers.loop_decorator import optimization_loop
import time
import os
from experiment_helpers.data_helpers import split_data
from experiment_helpers.saving_helpers import save_and_load_functions
from datasets import load_dataset
from overcomplete.sae import TopKSAE,QSAE, JumpSAE, BatchTopKSAE,losses
from overcomplete.sae.trackers import DeadCodeTracker
import numpy as np
import torch.nn.functional as F
from unet_autopsy import get_shape_dict
import json
from typing import Optional
from torchvision.transforms import Resize

#https://github.com/KempnerInstitute/overcomplete

parser=default_parser({
    "project_name":"sae",
    "src_dataset":"jbaker361/filler",
    "repo_id":"jlbaker361/sae-test"
})

KSAE="ksae"
JUMP="jump"
BATCHK="batch_k"
QUANTIZED="quantized"

parser.add_argument("--nb_concepts",type=int,default=10000,help="n concepts for SAE")
parser.add_argument("--sae_model",type=str,default=KSAE)
parser.add_argument("--model_layer",type=str,default="up_blocks.1.attentions.0")
parser.add_argument("--local_global_split",action="store_true",help="if yes, split the concepts into global and local components; global need to be the same across time and/or location")
parser.add_argument("--src_dir_list",nargs="*")
parser.add_argument("--dino",action="store_true",help="whether to use dino embeddings too")
parser.add_argument("--mask",action="store_true",help="whether to mask out irrelevant tensors")
parser.add_argument("--flatten",action="store_true",help="whether to do the whole image at once or do channel by channel")
parser.add_argument("--timestep",type=int,default=2,help="which timestep to use (probably doesn't matter)")
parser.add_argument("--hf_data",action="store_true")
parser.add_argument("--dino_coefficient",type=float,default=0.1)
parser.add_argument("--pooling",type=str,default="avg")
parser.add_argument("--threshold",type=float,default=0.5)
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7",help=" dreamshaper ")
parser.add_argument("--step",type=int,default=0)
parser.add_argument("--prefix",type=str,default="seg_ip_flickr")

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
    def __init__(self,src_dir_list:Optional[str|list[str]],checkpoint:str,step:int,model_layer:str,dino:bool,
                 use_mask:bool,limit:int,flatten:bool,pooling:str,threshold:float,
                 device):
        self.model_layer=model_layer
        self.src_dir_list=src_dir_list
        super().__init__()
        if type(src_dir_list)==str:
            src_dir_list=[src_dir_list]
        self.np_list=[]
        self.dino_list=[]
        self.mask_list=[]
        shape_dict=get_shape_dict(checkpoint,device)
        (b,c,h,w)=shape_dict[model_layer]
        self.h=h
        self.w=w
        self.c=c
        self.dino=dino
        self.mask=use_mask
        for src_dir in src_dir_list:
            self.np_list+=sorted([
                os.path.join(src_dir,str(step),f) for f in os.listdir(os.path.join(src_dir,str(step))) if f.endswith("npz")
            ],key=get_num)

            if dino:
                self.dino_list+=sorted([
                    os.path.join(src_dir,"dino",f) for f in os.listdir(os.path.join(src_dir,"dino")) if f.endswith("npy")
                ],key=get_num)
            if use_mask:
                self.mask_list+=sorted([
                    os.path.join(src_dir,"mask",f) for f in os.listdir(os.path.join(src_dir,"mask")) if f.endswith("npy")
                ],key=get_num)
        self.pool={
            "max":F.max_pool2d,
            "avg":F.avg_pool2d
        }[pooling]
        if use_mask:
            mask=torch.tensor(np.load(self.mask_list[0]))
            (m_h,m_w)=mask.size()
            self.kernel=m_h//h
        self.threshold=threshold
        self.flatten=flatten
        

    def __len__(self):
        return len(self.np_list)
    
    def __getitem__(self, index):
        ret={"act": torch.tensor(np.load(self.np_list[index])[self.model_layer][0])}
        (c,h,w)=ret["act"].size()
        num=get_num(self.np_list[index])
        if self.mask:
            mask=np.load(self.mask_list[index])
            ret["mask"] = torch.tensor(mask).unsqueeze(0)
            #print("mask path ",self.mask_list[index])

            ret["mask"]=Resize((h,w),)(ret["mask"])
            ret["mask"] = (ret["mask"] > self.threshold).to(torch.uint8)

            ret["act"]=ret["mask"]*ret["act"]
        if self.flatten:
            ret["act"]=ret["act"].flatten()
        else:
            ret["act"]=ret["act"].permute(1,2,0).flatten(0,1) # c,h,w -> (h,w,c) -> (hw,c)
        if self.dino:
            ret["dino"]=torch.tensor(np.load(self.dino_list[index])[0][0])
            if not self.flatten:
                ret["dino"]=ret["dino"].unsqueeze(-1).unsqueeze(-1).expand(-1,h,w) #384 -> 384,1,1 -> 384,h,w
                if self.mask:
                    ret["dino"]=ret["dino"]*ret["mask"]
                ret["dino"]=ret["dino"].permute(1,2,0).flatten(0,1) # 384,h,w -> (384,w,c) -> (hw,384)
            
            
        return ret
    

def main(args):
    api,accelerator,device=repo_api_init(args)
    
    sae_model_class={
        KSAE:TopKSAE,
        JUMP:JumpSAE,
        BATCHK:BatchTopKSAE,
        QUANTIZED:QSAE
    }[args.sae_model]
    
    criterion={
        KSAE:losses.mse_l1 #TODO: find losses for other models
    }[args.sae_model]

    dataset= LatentLocalDataset(args.src_dir_list,args.checkpoint,
                                args.step,args.model_layer,args.dino,args.mask,args.limit,args.flatten,args.pooling,
                                args.threshold,device)
    
    train_loader,test_loader,val_loader=split_data(dataset,0.95,args.batch_size)
    
    for batch in train_loader:
        break
    
    print("real activation size ",batch["act"].size())
    
    shape_dict=get_shape_dict(args.checkpoint,device)
    print(shape_dict[args.model_layer])
    (b,c,h,w)=shape_dict[args.model_layer]
    
    if args.flatten:
    
        sae_model=sae_model_class(c*h*w,args.nb_concepts,device=device)
    else:
        sae_model=sae_model_class(c,args.nb_concepts,device=device)
        
    
    params=[p for p in sae_model.parameters()]
    model_dict={
       "sae" :sae_model
    }
    
    if args.dino:
        dino=batch["dino"]
        print("dino size ",dino.size())
        if args.flatten:
            (b,dc)=dino.size() #(b,384hw)
        else:
            (b,hw,dc)=dino.size() #(b,hw,384)
        dino_sae_model=sae_model_class(dc,args.nb_concepts,device=device)
        params.extend([p for p in dino_sae_model.parameters()])
        model_dict["dino_sae"]=dino_sae_model
        
    act=batch["act"]
    if not args.flatten:
        act=act.flatten(0,1) #(b,hw,c) -> (bhw,c)
    with accelerator.autocast():
        z_pre, z, x_hat=sae_model(act.to(device))
    dead_tracker = DeadCodeTracker(z.size()[1], device)
    
    for t,name in zip([z_pre, z, x_hat],['z_pre', 'z', 'x_hat']):
        print(name,t.size())

    optimizer=torch.optim.AdamW(params,args.lr)
    
    if args.dino:
        dino_sae_model,optimizer,sae_model,train_loader,test_loader,val_loader = accelerator.prepare(dino_sae_model,optimizer,sae_model,train_loader,test_loader,val_loader)
    else:
        optimizer,sae_model,train_loader,test_loader,val_loader = accelerator.prepare(optimizer,sae_model,train_loader,test_loader,val_loader)

    step=str(args.step)
    save_subdir=os.path.join("sae_model")
    os.makedirs(save_subdir,exist_ok=True)
    save_subdir=os.path.join("sae_model",args.prefix)
    os.makedirs(save_subdir,exist_ok=True)
    save_subdir=os.path.join("sae_model",args.prefix,args.model_layer)
    os.makedirs(save_subdir,exist_ok=True)
    save_subdir=os.path.join("sae_model",args.prefix,args.model_layer,step)
    os.makedirs(save_subdir,exist_ok=True)
    


    def save(epoch: int):
        os.makedirs(save_subdir, exist_ok=True)

        weights_path = os.path.join(save_subdir, "weights.pt")
        config_path = os.path.join(save_subdir, "config.json")

        # save model weights
        torch.save(sae_model.state_dict(), weights_path)

        # optional DINO weights
        if args.dino:
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
            if args.dino:
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

    @optimization_loop(
        accelerator,train_loader,args.epochs,args.val_interval,args.limit,
        val_loader,test_loader,save,start_epoch
    )
    def batch_function(batch,training,helpful_dict):
        
        activations=batch["act"].to(device)
        if not args.flatten:
            activations=activations.flatten(0,1)
            
        if args.dino:
            dino=batch["dino"].to(device)
            if not args.flatten:
                dino=dino.flatten(0,1)
        if training:
            optimizer.zero_grad()
            models=[sae_model]
            if args.dino:
                models.append(dino_sae_model)
            with accelerator.accumulate(*models):
                with accelerator.autocast():
                    z_pre, z, x_hat=sae_model(activations)
                    
                    loss=criterion(activations, x_hat, z_pre, z, sae_model.get_dictionary())
                    
                    if args.dino: 
                        z_pre_dino, z_dino, x_hat_dino=dino_sae_model(dino)
                        
                        dino_loss=criterion(dino,x_hat_dino,z_pre_dino,z_dino,dino_sae_model.get_dictionary())
                        
                        difference=args.dino_coefficient*F.mse_loss(z_dino,z)
                        
                        loss+=difference+dino_loss
                        
                    
                    accelerator.backward(loss)
                    optimizer.step()
                    dead_tracker.update(z)
        else:
            with torch.no_grad():     
                z_pre, z, x_hat=sae_model(activations)
                    
                loss=criterion(activations, x_hat, z_pre, z, sae_model.get_dictionary())
                
                if args.dino: 
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
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")