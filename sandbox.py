from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, QWEN2_5_VL, VLM_IMAGE_TOKENS, Qwen2_VL_process_fn
from src.model.processor import LLAVA_QWEN2, FastVLM_process_fn
from src.utils import batch_to_device
from PIL import Image
import torch
import os

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def get_grid_size(model: MMEBModel, inputs):
    if model.model_backbone == LLAVA_QWEN2:
        vision_tower = model.encoder.get_vision_tower()
        vision_config = vision_tower.config
        patch_size = vision_config['image_cfg']['patch_size']
        grid_sizes = []
        for image in inputs['images']:
            if image is None:
                continue
            h, w = image.shape[-2:]
            grid_h = h // patch_size
            grid_w = w // patch_size
            grid_sizes.append((grid_h, grid_w))
        return grid_sizes
    
    elif model.model_backbone in [QWEN2_VL, QWEN2_5_VL]:
        vision_config = model.config.vision_config
        merge_size = vision_config.spatial_merge_size
        grid_sizes = []
        for shape in inputs['image_grid_thw']:
            if shape is None:
                continue
            h, w = shape[0, -2:]
            grid_h = (h // merge_size).item()
            grid_w = (w // merge_size).item()
            grid_sizes.append((grid_h, grid_w))
        return grid_sizes

# model_args = ModelArguments(
#     model_name='Qwen/Qwen2-VL-2B',
#     checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-2B',
#     pooling='last',
#     normalize=True,
#     model_backbone='qwen2_vl',
#     lora=True
# )
model_args = ModelArguments(
    model_name='apple/FastVLM-0.5B',
    pooling='last',
    normalize=True,
    model_backbone=LLAVA_QWEN2,
    lora=True
)
data_args = DataArguments()

processor = load_processor(model_args, data_args)
model = MMEBModel.build(model_args)
model = model.to('cuda', dtype=torch.bfloat16)
model.eval()

def disable_lora_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.lower().find("lora") != -1:
            for attr in ["dropout", "lora_dropout"]:
                if hasattr(m, attr):
                    d = getattr(m, attr)
                    if hasattr(d, "p"):
                        d.p = 0.0
disable_lora_dropout(model)
# Batch processing
processor_inputs = {
    "text": [f'{VLM_IMAGE_TOKENS[LLAVA_QWEN2]} Represent the given image with the following question: What is in the image',
          f'{VLM_IMAGE_TOKENS[LLAVA_QWEN2]} Represent the given image with the following question: What is in the image'],
    "images": [Image.open('example.jpg'),
            Image.open('example.jpg').resize((600, 1000))],
}
inputs = FastVLM_process_fn(
    processor_inputs,
    processor, 
    # square_padding=True
    )

inputs = batch_to_device(inputs, "cuda")
# with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#     qry_output = model(qry=inputs)["qry_reps"]


processor_inputs = {
    "text": ['A cat and a dog',  f'{VLM_IMAGE_TOKENS[LLAVA_QWEN2]} '+'A cat and a tiger'],
    "images": [None, Image.open('example.jpg')],
}
inputs = FastVLM_process_fn(
    processor_inputs,
    processor)
inputs = batch_to_device(inputs, "cuda")
# with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#     tgt_output = model(tgt=inputs)["tgt_reps"]
# print(model.compute_similarity(qry_output, tgt_output))

print(inputs['images'].shape)
print(inputs)
