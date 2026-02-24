import huggingface_hub
import torch
from experiment_helpers.gpu_details import print_details
print_details()

api=huggingface_hub.HfApi()

path=api.hf_hub_download("Wan-AI/Wan2.2-TI2V-5B",filename="Wan2.2_VAE.pth")

import os
os.symlink(os.path.join(os.getcwd(),"Wan2.2","wan"),"wan")

from wan.text2video import WanT2V
from wan.configs import WAN_CONFIGS

task="ti2v-5B'"
rank=4

device=torch.cuda.get_device_name() if torch.cuda.device_count() > 0 else "cpu"

cfg = WAN_CONFIGS[task]
wan_t2v = WanT2V(
            config=cfg,
            checkpoint_dir=path,
            device_id=device,
            rank=rank,
            #t5_fsdp=args.t5_fsdp,
            #dit_fsdp=args.dit_fsdp,
            #use_sp=(args.ulysses_size > 1),
            t5_cpu=True,
            convert_model_dtype=True,
        )

print("all done!")