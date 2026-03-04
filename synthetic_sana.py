import os
import time
import torch
from accelerate import Accelerator
from datasets import Dataset, Features, Image, Value
from diffusers import SanaSprintPipeline
from experiment_helpers.gpu_details import print_details

print_details()

# ---------------------------
# CONFIG
# ---------------------------
subject_path = "subjects.txt"
style_path = "styles.txt"
repo_id = "jlbaker361/synthetic-sana"
limit = 30
seed = 42
num_inference_steps = 2
gpu_batch_size = 4
cpu_batch_size = 1
# ---------------------------

# Detect device
accelerator = Accelerator()
device = accelerator.device
is_cpu = device.type == "cpu"

if accelerator.is_main_process:
    print(f"Running on device: {device}")

# Load prompts
with open(subject_path, "r") as f:
    subject_list = [s.strip() for s in f.readlines()]

with open(style_path, "r") as f:
    style_list = [s.strip() for s in f.readlines()]

# Build prompt list
all_prompts = []
for sub in subject_list:
    for sty in style_list:
        all_prompts.append((sub, sty, f"{sub}, {sty}"))
        if len(all_prompts) >= limit:
            break
    if len(all_prompts) >= limit:
        break

# ---------------------------
# PIPELINE
# ---------------------------
dtype = torch.float32 if is_cpu else torch.bfloat16
batch_size = cpu_batch_size if is_cpu else gpu_batch_size

pipe = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=dtype
).to(device)

generator = torch.Generator(device=device).manual_seed(seed)

# ---------------------------
# DISTRIBUTED OR SIMPLE LOOP
# ---------------------------
if not is_cpu:


    local_images = []
    local_subjects = []
    local_styles = []
    local_prompts = []

    start = time.time()
    with torch.no_grad():
        for i in range(0, len(all_prompts), batch_size):
            batch=all_prompts[i:i+batch_size]
            prompts=[b[2] for b in batch ]
            subjects=[b[0] for b in batch]
            styles=[b[1] for b in batch]
            images = pipe(list(prompts), num_inference_steps=num_inference_steps, generator=generator).images

            local_images.extend(images)
            local_subjects.extend(subjects)
            local_styles.extend(styles)
            local_prompts.extend(prompts)

    end = time.time()
    accelerator.print(f"Process {accelerator.process_index} generated {len(local_images)} images in {end-start:.2f}s")

    # Gather across GPUs
    all_images = accelerator.gather_for_metrics(local_images)
    all_subjects = accelerator.gather_for_metrics(local_subjects)
    all_styles = accelerator.gather_for_metrics(local_styles)
    all_prompts = accelerator.gather_for_metrics(local_prompts)

else:
    # CPU mode: simple loop
    all_images = []
    all_subjects = []
    all_styles = []
    all_prompts_cpu = []

    start = time.time()
    with torch.no_grad():
        for sub, sty, prompt in all_prompts:
            image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
            all_images.append(image)
            all_subjects.append(sub)
            all_styles.append(sty)
            all_prompts_cpu.append(prompt)
    end = time.time()
    print(f"CPU generated {len(all_images)} images in {end-start:.2f}s")

    all_prompts = all_prompts_cpu

# ---------------------------
# PUSH TO HUB
# ---------------------------
if accelerator.is_main_process:
    print(f"Total images collected: {len(all_images)}")

    features = Features({
        "image": Image(),
        "subject": Value("string"),
        "style": Value("string"),
        "prompt": Value("string"),
    })

    dataset = Dataset.from_dict(
        {
            "image": all_images,
            "subject": all_subjects,
            "style": all_styles,
            "prompt": all_prompts,
        },
        features=features,
    )

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    dataset.push_to_hub(
        repo_id,
        commit_message=f"Synthetic Sana dataset | {len(all_images)} images | steps={num_inference_steps}",
        revision=f"v_{timestamp}",
    )

    print("Upload complete.")