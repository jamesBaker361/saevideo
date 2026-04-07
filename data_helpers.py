from PIL import Image
from torch.utils.data import DataLoader,Dataset
from diffusers.image_processor import VaeImageProcessor
import os
import json

class PersonaDataset(Dataset):
    def __init__(self,subset:str,size:tuple[int],keyword:bool=True):
        super().__init__()
        self.image_processor= VaeImageProcessor()
        self.subset=subset
        self.size=size
        self.keyword=keyword
        
        with open(os.path.join("pcs_dataset","info.json")) as f:
            mapping=json.load(f)
        
        self.path_list=[
            
        ]
        
        self.text_list=[
            
        ]
        self.keyword_list=[]
        if subset=="subject":
            mapping_sub=mapping["subjects"]
        

            for k,v in mapping_sub["subject_with_cls"].items():
                if v in mapping_sub["live_subjects"]:
                    prompt_list= mapping_sub["prompt_live"]
                else:
                    prompt_list=mapping_sub["prompt_object"]
                for prompt in prompt_list:
                    self.text_list.append(prompt.replace("{0} {1}",v))
                    self.path_list.append(os.path.join("pcs_dataset","subjects",k,"00.jpg"))
                    self.keyword_list.append(k)
            
        elif subset=="face":
            mapping_face=mapping["face"]
            
            prompt_list=mapping_face["prompt_accessory"]+mapping_face["prompt_context"]+mapping_face["prompt_action"]+mapping_face["prompt_style"]
            
            for k,v in mapping_face["id_with_gender"].items():
            
                for prompt in prompt_list:
                    if self.keyword:
                        self.text_list.append(prompt.replace("{0} {1}",v))
                    else:
                        self.text_list.append(prompt.replace("{0} {1}"," "))
                    self.path_list.append(os.path.join("pcs_dataset","face",k,"face.jpg"))
                    self.keyword_list.append(k)
            
        elif subset=="style":
            mapping_style=mapping["style"]
            
            
    def __len__(self):
        return len(self.text_list)
    
    def __getitem__(self, index):
        return {
            "image":self.image_processor.preprocess(
                Image.open(self.path_list[index]).resize(self.size)
            ),
            "image_pil":Image.open(self.path_list[index]).resize(self.size),
            "text":self.text_list[index],
            "keyword":self.keyword_list[index]
        }