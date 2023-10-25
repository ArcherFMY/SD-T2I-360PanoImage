# SD-T2I-360PanoImage
repository for 360 panorama image generation based on Stable Diffusion

# ![a living room](data/a-living-room.png "a living room")
# ![the mountains](data/the-mountains.png "the mountains")
# ![the times square](data/the-times-square.png "the times square")


## Requirements
- torch
- torchvision
- torchaudio
- diffusers
- accelerate
- xformers
- triton
- transformers
- realesrgan


## Installation
```
git clone https://github.com/ArcherFMY/SD-T2I-360PanoImage.git
cd SD-T2I-360PanoImage
pip install -r requirements.txt
```

## Getting Started
### Download Models
Download models from [Baidu Disk](https://pan.baidu.com/s/1i_ypdWHknp2kqbjl0_zAuw?pwd=w2vr). Unzip `models.zip` into the root directory of the project.
```
${ROOT}  
|-- data  
|   |-- a-living-room.png
|   |...
|-- models  
|   |-- sd-base
|   |-- sr-base
|   |-- sr-control
|   |-- RealESRGAN_x2plus.pth
|-- txt2panoimg
|...
```
Or download the models from [hugging face](https://huggingface.co/archerfmy0831/sd-t2i-360panoimage)

### Inference
```
import torch
from txt2panoimage import Text2360PanoramaImagePipeline

prompt = 'The living room'
input = {'prompt': prompt, 'upscale': False}
model_id = './models'
txt2panoimg = Text2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)
output = txt2panoimg(input)

output.save('result.png')
```
see more in `demo.py`

### Use in ModelScope
see [here](https://www.modelscope.cn/models/damo/cv_diffusion_text-to-360panorama-image_generation/summary) for more information.

## License

This code is released under the Apache License 2.0 (refer to the LICENSE file for details).



