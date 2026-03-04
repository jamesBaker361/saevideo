import torch
from diffusers import DiffusionPipeline

class HookWrapper:
    def __init__(self,pipe:DiffusionPipeline, layers:list[str]):
        self.pipe=pipe
        self.layers=layers
        self.activations = {}
        
        if getattr(pipe,"unet",None) != None:
            net=pipe.unet
        elif getattr(pipe,"transformer",None) != None:
            net=pipe.transformer

        def save_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        for name, module in net.named_modules():
            if name in layers:
                module.register_forward_hook(save_hook(name))
                
    def __call__(self,*args,**kwargs):
        self.activations={}
        result=self.pipe(*args,**kwargs)
        return result,self.activations
    
if __name__=="__main__":
    pipe=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    hw=HookWrapper(pipe,['down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj'])
    hw("hello",**{"num_inference_steps":2})