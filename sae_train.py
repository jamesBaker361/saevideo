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
parser.add_argument("--src_dir",type=str,default="features_SimianLuo_LCM_Dreamshaper_v7_mini")
parser.add_argument("--dino",action="store_true",help="whether to use dino embeddings too")
parser.add_argument("--mask",action="store_true",help="whether to mask out irrelevant tensors")
parser.add_argument("--flatten",action="store_true",help="whether to do the whole image at once or do channel by channel")
parser.add_argument("--timestep",type=int,default=2,help="which timestep to use (probably doesn't matter)")
parser.add_argument("--hf_data",action="store_true")
parser.add_argument("--dino_coefficient",type=float,default=0.1)

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self,hf_dataset,model_layer):
        super().__init__()
        self.data=load_dataset(hf_dataset)
        self.model_layer=model_layer
        
    def __len__(self):
        return len(self.data[self.model_layer])
    
    def __getitem__(self, index):
        return torch.tensor(self.data[self.model_layer][index])
    
class LatentLocalDataset(torch.utils.data.Dataset):
    def __init__(self,src_dir:str,step:int,model_layer:str,dino_mask:bool):
        self.model_layer=model_layer
        self.src_dir=src_dir
        super().__init__()
        self.np_list=[
            os.path.join(src_dir,str(step),f) for f in os.listdir(os.path.join(src_dir,str(step))) if f.endswith("npz")
        ]
        self.dino_mask=dino_mask
        if dino_mask:
            self.dino_list=[
                os.path.join(src_dir,"dino",f) for f in os.listdir(os.path.join(src_dir,"dino")) if f.endswith("np")
            ]
            self.mask_list=[
                os.path.join(src_dir,"mask",f) for f in os.listdir(os.path.join(src_dir,"mask")) if f.endswith("np")
            ]

    def __len__(self):
        return len(self.np_list)
    
    def __getitem__(self, index):
        ret={"act": torch.tensor(np.load(self.np_list[index])[self.model_layer])}
        if self.dino_mask:
            ret["dino"]=torch.tensor(np.load(self.dino_list[index]))
            ret["mask"]=torch.tensor(np.load(self.mask_list[index]))
            
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
    }

    dataset= LatentLocalDataset(args.src_dir,0,args.model_layer,False)
    
    train_loader,test_loader,val_loader=split_data(dataset,0.95,args.batch_size)
    
    for batch in train_loader:
        break
    
    activations=batch["act"]
    act_size=activations.size()
        
    print("activation size ",act_size)
    if args.flatten:
        activations=activations.flatten(1)
    else:
        activations=activations.view(0,2,3,1).flatten(0,2)
        
    act_size=activations.size()
    
    (b,c,h,w)=act_size
        
    print("activation size ",act_size)
    
    sae_model=sae_model_class(activations.size()[1:],args.nb_concepts,device=device)
    
    params=[p for p in sae_model.parameters()]
    model_dict={
       "sae" :sae_model
    }
    
    if args.dino:
        dino=batch["dino"]
        dino_sae_model=sae_model_class(dino.size()[1:],args.nb_concepts,device=device)
        params.extend([p for p in dino_sae_model.parameters()])
        model_dict["dino_sae"]=dino_sae_model
    
    z_pre, z, x_hat=sae_model(batch)
    dead_tracker = DeadCodeTracker(z.size()[1], device)
    
    for t,name in zip([z_pre, z, x_hat],['z_pre', 'z', 'x_hat']):
        print(name,t.size())

    optimizer=torch.optim.AdamW(params,args.lr)
    
    if args.dino:
        dino_sae_model,optimizer,sae_model,action_encoder,train_loader,test_loader,val_loader = accelerator.prepare(dino_sae_model,optimizer,sae_model,action_encoder,train_loader,test_loader,val_loader)
    else:
        optimizer,sae_model,action_encoder,train_loader,test_loader,val_loader = accelerator.prepare(optimizer,sae_model,action_encoder,train_loader,test_loader,val_loader)

    save_subdir=os.path.join(args.save_dir,args.repo_id)
    os.makedirs(save_subdir,exist_ok=True)
    
    save,load=save_and_load_functions(
        model_dict
        ,save_subdir,api,args.repo_id)
    
    start_epoch=load()

    @optimization_loop(
        accelerator,train_loader,args.epochs,args.val_interval,args.limit,
        val_loader,test_loader,save,start_epoch
    )
    def batch_function(batch,training,helpful_dict):
        if training:
            activations=batch["act"]
            if args.dino:
                dino=batch["dino"]
            if args.mask:
                mask=batch["mask"].unsqueeze(1)
                activations=activations*mask
                
            if args.flatten:
                activations=activations.flatten(1)
            else:
                activations=activations.view(0,2,3,1).flatten(0,2)
                

            optimizer.zero_grad()
            with accelerator.accumulate(params):
                z_pre, z, x_hat=sae_model(activations)
                
                loss=criterion(activations, x_hat, z_pre, z, sae_model.get_dictionary())
                
                if args.dino: 
                    if args.flatten:
                        pass #dino=dino.flatten(1)
                    else:
                        dino=dino[:, :, None, None].expand(-1, -1, h, w)
                    
                        if args.mask:
                            dino=dino*mask
                        dino=dino.view(0,2,3,1).flatten(0,2)
                        
                    z_pre_dino, z_dino, x_hat_dino=dino_sae_model(dino)
                    
                    dino_loss=criterion(dino,x_hat_dino,z_pre_dino,z_dino)
                    
                    difference=args.dino_coefficient*F.mse_loss(z_dino,z)
                    
                    loss+=difference+dino_loss
                    
                    
                
                accelerator.backward(loss)
                optimizer.step()
                    
                
                dead_tracker.update(z)
        else:
            with torch.no_grad():     
                loss=torch.tensor(0) #placeholder
                
        return loss.cpu().detach().numpy()
        
        
    batch_function()
    
    
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