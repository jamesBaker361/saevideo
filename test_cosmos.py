import torch
from diffusers import CosmosTextToWorldPipeline
from custom_pipeline_cosmos_text2world import CustomCosmosTextToWorldPipeline
from diffusers.utils import export_to_video
from experiment_helpers.gpu_details import print_details
print_details()

model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
pipe = CustomCosmosTextToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
try:
    pipe.to("cuda")
except RuntimeError:
    pass

prompt = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."

output = pipe(prompt=prompt).frames[0]
export_to_video(output, "output.mp4", fps=30)


print("all done!")