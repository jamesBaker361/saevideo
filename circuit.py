from diffusers import DiffusionPipeline
from hook_wrapper import HookWrapper
import argparse
import accelerate
from torch.nn import Conv2d

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7")

args=parser.parse_args()

accelerator =accelerate.Accelerator()


pipe =DiffusionPipeline.from_pretrained(args.checkpoint).to(accelerator.device)

conv_layers=[]

print(type(pipe))
print(dir(pipe))
unet=pipe.unet
for name,module in unet.named_modules():
    if type(module)==Conv2d:
        conv_layers.append(module)
        
        print(name,module.in_channels,module.out_channels,module.weight.size())