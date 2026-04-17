import os
os.environ["TQDM_DISABLE"] = "1"
import time
import torch
from accelerate import Accelerator
from datasets import Dataset, Features, Image, Value
from diffusers import SanaSprintPipeline
from experiment_helpers.gpu_details import print_details
import sys
from datasets import load_dataset
import argparse

print_details()

parser=argparse.ArgumentParser()
parser.add_argument("--folder",type=str,default="synthetic-sana2")
parser.add_argument("--dataset",type=str,default="txt")

args=parser.parse_args()


import torch
print("Torch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

x = torch.randn(1, 3, 64, 64)
conv = torch.nn.Conv2d(3, 8, 3)

try:
    y = conv(x)
    print("Conv2d works on CPU")
except Exception as e:
    print("CPU conv failed:", e)

if torch.cuda.is_available():
    conv = conv.cuda()
    x = x.cuda()
    try:
        y = conv(x)
        print("Conv2d works on CUDA")
    except Exception as e:
        print("CUDA conv failed:", e)

# ---------------------------
# CONFIG
# ---------------------------
subject_path = "subjects.txt"
style_path = "styles.txt"
repo_id = "jlbaker361/synthetic-sana2"
limit = 10000000
seed = 42
num_inference_steps = 2
gpu_batch_size = 2
cpu_batch_size = 1
# ---------------------------

# Detect device
accelerator = Accelerator()
device = accelerator.device
is_cpu = device.type == "cpu"

if accelerator.is_main_process:
    print(f"Running on device: {device}")

if args.dataset=='txt':
    # Load prompts
    with open(subject_path, "r") as f:
        subject_list = [s.strip() for s in f.readlines()]

    with open(style_path, "r") as f:
        style_list = [s.strip() for s in f.readlines()]

    # Build prompt list
    all_prompts = []
    for sub in subject_list:
        for sty in style_list:
            all_prompts.append(f"{sub}, {sty}")
            if len(all_prompts) >= limit:
                break
        if len(all_prompts) >= limit:
            break
elif args.dataset=='flickr':
    data=load_dataset("AnyModal/flickr30k",split="train")
    all_prompts=[row["alt_text"][0] for row in data]
elif args.dataset=='laion':
    data=load_dataset("laion/laion2B-en-aesthetic",split="train")
    all_prompts=[row["TEXT"] for row in data]
else:
    print("unrezognized datasset",args.dataset)

# ---------------------------
# PIPELINE
# ---------------------------
dtype = torch.float32 if is_cpu else torch.float16
batch_size = cpu_batch_size if is_cpu else gpu_batch_size

pipe = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=dtype
).to(device)

generator = torch.Generator(device=device).manual_seed(seed)
count=0

folder=args.folder
os.makedirs(folder,exist_ok=True)
config="config.csv"
count=len([f for f in os.listdir(folder) if f.endswith("jpg")])
        
print("count ",count,f"/{len(all_prompts)*batch_size}")
with open(os.path.join(folder,config),"a",buffering=1) as file:
    

# ---------------------------
# DISTRIBUTED OR SIMPLE LOOP
# ---------------------------
    if not is_cpu:


        local_prompts = []

        start = time.time()
        with torch.no_grad():
            for i in range(0, len(all_prompts), batch_size):
                if i<count:
                    continue
                
                prompts=all_prompts[i:i+batch_size]
                images = pipe(list(prompts), num_inference_steps=num_inference_steps, generator=generator,height=256,width=256).images

                local_prompts.extend(prompts)
                if i%100==0:
                    end = time.time()
                    accelerator.print(f"Process {accelerator.process_index} generated {len(local_prompts)+count} images in {end-start:.2f}s")

                for img,prompt in zip(images,prompts):
                    path=os.path.join(folder,f"{count}.jpg")
                    img.save(path)
                    file.write(f"{path},{prompt}\n")
                    count+=1
                        
                

        end = time.time()
        accelerator.print(f"Process {accelerator.process_index} generated {len(local_prompts)} images in {end-start:.2f}s")

    else:
        # CPU mode: simple loop
        start = time.time()
        with torch.no_grad():
            for i, prompt in enumerate(all_prompts):
                if i < count:
                    continue
                image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
                path=os.path.join(folder,f"{count}.jpg")
                image.save(path)
                file.write(f"{path},{prompt}\n")
                count+=1
        end = time.time()
        print(f"CPU generated {count} images in {end-start:.2f}s")