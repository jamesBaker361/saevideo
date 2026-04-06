import torch
from diffusers import DiffusionPipeline,UNet2DConditionModel
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
    def __init__(self, underlying:torch.nn.Module,weight:float,name:str,keys:list[int]=[],cross_attn_cache:bool=False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.underlying=underlying
        self.output=None
        self.weight=weight
        self.name=name
        self.keys=keys
        self.cross_attn_cache=cross_attn_cache
        
    def forward(self, *args, **kwargs):
        result = self.underlying(*args, **kwargs)

        def blend(x):
            (b,c,h,w)=x.size()

            if self.output is None:
                return x
            (b,c,h,w)=x.size()
            
            if self.keys!=[] and self.cross_attn_cache:
                cached={}
                for name,module in self.underlying.named_children():
                    for n in [1,2]:
                        for a in ["k,q,v"]:
                            target=f"attn{n}.to_{a}"
                            if name==target:
                                cached[target]=module.cached_output
            
            if len(self.output.size())==2:
                (b,c)=self.output.size()
                self.output=self.output.view(1,c,1,1).expand(b,c,h,w)
            out = self.output.to(x.device, x.dtype)
            return (self.weight * out) + ((1 - self.weight) * x)

        if isinstance(result, tuple):
            main = result[0]
            main = blend(main)
            
            return (main, *result[1:])
        else:
            return blend(result)
        
        
def set_by_path(obj, path: str, new_module):
    parts = path.split(".")
    parent = obj

    # Traverse to parent
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last = parts[-1]

    # Replace final node
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)
        

def getattr_named(unet:UNet2DConditionModel,target_name:str):
    for name, module in unet.named_modules():
        if name == target_name:
            return module
    return None

class HookPipe:
    def __init__(self, pipe, layers, sae_dict, shape_dict, weight):
        self.pipe = pipe
        self.layers = set(layers)  # faster lookup
        self.sae_dict = sae_dict
        self.shape_dict = shape_dict
        self.weight = weight

        # IMPORTANT: collect matches first
        matches = []
        for name, module in self.pipe.unet.named_modules():
            if name in self.layers:
                matches.append((name, module))

        # Now modify AFTER iteration
        for name, module in matches:
            # Optional: avoid wrapping twice
            if isinstance(module, MonkeyModule):
                continue

            wrapped = MonkeyModule(module, weight,name)
            set_by_path(self.pipe.unet, name, wrapped)
            print(f"set {name}")
    
    def forward(self,sae_src_dict:dict[torch.Tensor],*args,**kwargs):
        for layer in self.layers:  #,self.sae_dict,sae_src_dict):
            sae=self.sae_dict[layer]
            src=sae_src_dict[layer]
            output=sae.decode(src)
            
            getattr_named(self.pipe.unet,layer).output=output
        
        return self.pipe(*args,**kwargs)    
    
    
class HookUNet:
    def __init__(self,unet:UNet2DConditionModel,layers:list, sae_dict:dict, weight:float,cross_attn_cache:bool=False):
        
        self.unet=unet
        self.layers = set(layers)  # faster lookup
        self.sae_dict = sae_dict
        self.weight = weight
        
        def save_hook(name):
            def hook(module, input, output):
                if type(output)==tuple:
                    output=output[0]
                setattr(module,"cached_output",output)
            return hook
        
        matches = []
        for name, module in self.unet.named_modules():
            if name in self.layers:
                matches.append((name, module))
                
            if cross_attn_cache:
                if name.find("to_q") !=-1 or name.find("to_k") !=-1 or name.find("to_v") !=-1:
                    module.register_forward_hook(save_hook(name))

        # Now modify AFTER iteration
        for name, module in matches:
            # Optional: avoid wrapping twice
            if isinstance(module, MonkeyModule):
                continue

            wrapped = MonkeyModule(module, weight,name)
            set_by_path(self.unet, name, wrapped)
            print(f"set {name}")
        self.cross_attn_cache=cross_attn_cache
        
            
            
    def forward(self,sae_src_dict:dict[torch.Tensor],*args,**kwargs):
        for layer in self.layers:  #,self.sae_dict,sae_src_dict):
            sae=self.sae_dict[layer]
            src=sae_src_dict[layer]
            output=sae.decode(src)
            
            getattr_named(self.unet,layer).output=output
        
        return self.unet(*args,**kwargs)   
    
    
if __name__=="__main__":
    pipe=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    hw=HookWrapper(pipe,['down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj'])
    r,act=hw("hello",**{"num_inference_steps":2})
    print(act['down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj'].size())
    