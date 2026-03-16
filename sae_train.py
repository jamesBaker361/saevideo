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
parser.add_argument("--use_dino_mask",action="store_true",help="whether to use dino embeddings and mask too")
parser.add_argument("--timestep",type=int,default=2,help="which timestep to use (probably doesn't matter)")
parser.add_argument("--hf_data",action="store_true")

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
    def __init__(self,src_dir,step,model_layer):
        self.model_layer=model_layer
        super().__init__()
        self.np_list=[
            os.path.join(src_dir,str(step),f) for f in os.listdir(os.path.join(src_dir,str(step))) if f.endswith("npz")
        ]

    def __len__(self):
        return len(self.np_list)
    
    def __getitem__(self, index):
        return torch.tensor(np.load(self.np_list[index])[self.model_layer])
    

def main(args):
    api,accelerator,device=repo_api_init(args)
    
    sae_model={
        KSAE:TopKSAE,
        JUMP:JumpSAE,
        BATCHK:BatchTopKSAE,
        QUANTIZED:QSAE
    }[args.sae_model]
    
    criterion={
        KSAE:losses.mse_l1 #TODO: find losses for other models
    }

    dataset= LatentDataset(args.src_dataset,)
    
    train_loader,test_loader,val_loader=split_data(dataset,0.8,args.batch_size)
    
    for batch in train_loader:
        break
    
    z_pre, z, x_hat=sae_model(batch)
    dead_tracker = DeadCodeTracker(z.shape[1], device)

    save_subdir=os.path.join(args.save_dir,args.repo_id)
    os.makedirs(save_subdir,exist_ok=True)

    params=sae_model.parameters()

    optimizer=torch.optim.AdamW(params,args.lr)
    
    optimizer,sae_model,action_encoder,train_loader,test_loader,val_loader = accelerator.prepare(optimizer,sae_model,action_encoder,train_loader,test_loader,val_loader)

    
    save,load=save_and_load_functions({
       "sae" :sae_model
    },save_subdir,api,args.repo_id)
    
    start_epoch=load()

    @optimization_loop(
        accelerator,train_loader,args.epochs,args.val_interval,args.limit,
        val_loader,test_loader,save,start_epoch
    )
    def batch_function(batch,training,helpful_dict):
        if training:
            if args.local_global_split:
                pass #gotta figure this out
            else:
                batch=batch.to(device)
                optimizer.zero_grad()
                with accelerator.accumulate(params):
                    z_pre, z, x_hat=sae_model(batch)
                    
                    loss=criterion(batch, x_hat, z_pre, z, sae_model.get_dictionary())
                    accelerator.backward(loss)
                    optimizer.step()
                    
                
                dead_tracker.update(z)
        else:
            with torch.no_grad():     
                z_pre, z, x_hat=sae_model(batch)  
                loss=criterion(batch, x_hat, z_pre, z, sae_model.get_dictionary())
                
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