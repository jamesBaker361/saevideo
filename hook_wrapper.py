import torch
from diffusers import DiffusionPipeline
from overcomplete import SAE

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
                if name not in self.activations:
                    self.activations[name]=[]
                if type(output)==tuple:
                    output=output[0]
                try:
                    self.activations[name].append(output.detach().cpu())
                except:
                    print(type(module),type(output))
            return hook

        for name, module in net.named_modules():
            if name in layers:
                module.register_forward_hook(save_hook(name))
                
    def __call__(self,*args,**kwargs):
        self.activations={}
        result=self.pipe(*args,**kwargs)
        return result,self.activations
    
class MonkeyModule(torch.nn.Module):
    def __init__(self, underlying:torch.nn.Module,weight:float,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.underlying=underlying
        self.output=None
        self.weight=weight
        
    def forward(self,*args,**kwargs):
        result=self.underlying(*args,**kwargs)
        if type(result) is tuple:
            residual,result=result
            result=(self.weight*self.output)+((1-self.weight)*result)
            return (residual,result)
        else:
            result=(self.weight*self.output)+((1-self.weight)*result)
            return result
        

class HookForward:
    def __init__(self,pipe:DiffusionPipeline, layers:list[str],sae_list:list[SAE],shape_list:list,weight:float):
        self.pipe=pipe
        self.layers=layers
        self.sae_list=sae_list
        
        
        for l in self.layers:
            if self.pipe.unet.getattr(l,None) is not None:
                self.pipe.unet.setattr(l,MonkeyModule(self.pipe.unet.getattr(l),weight))
    
    def forward(self,sae_src_list:list[torch.Tensor],*args,**kwargs):
        for layer,sae,src in zip(self.layers,self.sae_list,sae_src_list):
            output=sae.decode(src)
            getattr(self.pipe.unet,layer).output=output
        
        return self.pipe(*args,**kwargs)    
    
    
    
if __name__=="__main__":
    pipe=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    hw=HookWrapper(pipe,['down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj'])
    r,act=hw("hello",**{"num_inference_steps":2})
    print(act['down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj'].size())
    