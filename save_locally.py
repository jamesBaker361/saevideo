import datasets

import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.argprint import print_args
import torch
import datasets

import time
import torch.nn.functional as F

from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init

parser=default_parser()
parser.add_argument("--dataset_name",type=str,default="",required=True)
parser.add_argument("--dest_dir",type=str,default="",required=True)
            

def main(args):
    dataset_name:str=args.dataset_name
    dest_dir:str=args.dest_dir
    
    hf_data=datasets.load_dataset(dataset_name,split="train") #ILSVRC/imagenet-1k or huggan/wikiart or ares1123/celebrity_dataset or huggan/pokemon or HuggingFaceM4/NoCaps or biglam/european_art or rafaelpadilla/coco2017
    hf_data=hf_data.cast_column("image",datasets.Image())
    os.makedirs(dest_dir,exist_ok=True)
    for r,row in enumerate(hf_data):
        img=row["image"]
        path=os.path.join(dest_dir,f"{r}.jpg")
        img.save(path)

        


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print_args(args)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")