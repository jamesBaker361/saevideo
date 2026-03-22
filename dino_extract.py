import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
dino_model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

def get_last_hidden_states(image,dino_processor,dino_model)->torch.Tensor:
    inputs = dino_processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = dino_model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states
