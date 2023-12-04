import torch
from diffusers.utils import load_image
from img2panoimg import Image2360PanoramaImagePipeline

image = load_image("./data/i2p-image.jpg").resize((512, 512))
mask = load_image("./data/i2p-mask.jpg")

prompt = 'The office room'

# for <16GB gpu
input = {'prompt': prompt, 'image': image, 'mask': mask, 'upscale': False}

# for >16GB gpu (24GB at least)
# the similarity with the input image is poor because of the super-resolution steps. It should be improved.
# input = {'prompt': prompt, 'image': image, 'mask': mask, 'upscale': True}

model_id = 'models'
img2panoimg = Image2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)
output = img2panoimg(input)
output.save('result.png')