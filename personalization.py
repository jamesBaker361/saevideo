'''
given image (persona dataset)
extract dino features
dino features to SAE features
at inference, find layer, replace its activations with SAE features put through decoder or sum or average them

OR train tensors that we add to the SAEs instead of dreamboothing it  -that should be somewhere else

'''

import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
import torch

import time
import torch.nn.functional as F

from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init

parser=default_parser()
parser.add_argument("--layers",nargs="*")
parser.add_argument("--sae_weights",nargs="*",help="list of paths for saes to load from")
parser.add_argument("--hidden_dim",nargs='*',help=" hidden dim of sae, if len = 1 then we default to all of them being the one thing")
            

def main(args):
    api,accelerator,device=repo_api_init(args)


        


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