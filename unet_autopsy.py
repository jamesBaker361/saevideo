from diffusers import DiffusionPipeline
from hook_wrapper import HookWrapper
import argparse
import accelerate

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7")

args=parser.parse_args()

accelerator =accelerate.Accelerator()


pipe =DiffusionPipeline.from_pretrained(args.checkpoint).to(accelerator.device)

print(type(pipe))
print(dir(pipe))
unet=pipe.unet
for name,module in unet.named_modules():
    print(name,type(module))
    
names=[name for name,module in unet.named_modules()]

hw=HookWrapper(pipe,names)

_,act=hw("image",num_inference_steps=12, height=128,width=128)

for k,v in act.items():
    print(k,v.size())