import torch
from txt2panoimg import Text2360PanoramaImagePipeline

prompt = 'The living room'
# for <16GB gpu
input = {'prompt': prompt, 'upscale': False}

# for >16GB gpu (24GB at least)
# input = {'prompt': prompt, 'upscale': True}

model_id = 'models'
txt2panoimg = Text2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)
output = txt2panoimg(input)
output.save('result.png')