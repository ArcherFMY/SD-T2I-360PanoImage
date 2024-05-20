# Copyright © Alibaba, Inc. and its affiliates.
# The implementation here is modifed based on diffusers.StableDiffusionPipeline,
# originally Apache 2.0 License and public available at
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

import copy
import inspect
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import os
import torch
import torch.nn.functional as F
from diffusers import (AutoencoderKL, DiffusionPipeline, ControlNetModel, UNet2DConditionModel)
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
try:
    from diffusers.models.autoencoders.vae import DecoderOutput
except:
    from diffusers.models.vae import DecoderOutput
from diffusers.models.controlnet import ControlNetOutput
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (PIL_INTERPOLATION, deprecate, is_accelerate_available,
                             is_accelerate_version, logging,
                             replace_example_docstring)
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import PIL
import numpy as np


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import EulerAncestralDiscreteScheduler
        >>> from txt2panoimage.pipeline_base import StableDiffusionBlendExtendPipeline
        >>> model_id = "models/sd-base"
        >>> pipe = StableDiffusionBlendExtendPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> pipe.vae.enable_tiling()
        >>> pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        >>> # remove following line if xformers is not installed
        >>> pipe.enable_xformers_memory_efficient_attention()
        >>> pipe.enable_model_cpu_offload()
        >>> prompt = "a living room"
        >>> image = pipe(prompt).images[0]
        ```
"""

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [['', 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(pipe: DiffusionPipeline, prompt: List[str],
                             max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning(
            'Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples'
        )
    return tokens, weights


def pad_tokens_and_weights(tokens,
                           weights,
                           max_length,
                           bos,
                           eos,
                           pad,
                           no_boseos_middle=True,
                           chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [
            bos
        ] + tokens[i] + [pad] * (max_length - 1 - len(tokens[i]) - 1) + [eos]
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (
                max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2):min(
                        len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe: DiffusionPipeline,
    text_input: torch.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2):(i + 1)
                                          * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            text_embedding = pipe.text_encoder(text_input_chunk)[0]

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        text_embeddings = pipe.text_encoder(text_input)[0]
    return text_embeddings


def get_weighted_text_embeddings(
    pipe: DiffusionPipeline,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        pipe (`DiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.tokenizer.model_max_length
                  - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(
            pipe, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(
                pipe, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [
            token[1:-1] for token in pipe.tokenizer(
                prompt, max_length=max_length, truncation=True).input_ids
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1] for token in pipe.tokenizer(
                    uncond_prompt, max_length=max_length,
                    truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length,
                         max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length
                  - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    pad = getattr(pipe.tokenizer, 'pad_token_id', eos)
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(
        prompt_tokens, dtype=torch.long, device=pipe.device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(
            uncond_tokens, dtype=torch.long, device=pipe.device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(
        prompt_weights,
        dtype=text_embeddings.dtype,
        device=text_embeddings.device)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(
            uncond_weights,
            dtype=uncond_embeddings.dtype,
            device=uncond_embeddings.device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(
            text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(
            text_embeddings.dtype)
        text_embeddings *= (previous_mean
                            / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(
                uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(
                uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean
                                  / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings, None


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (
        1 - guidance_rescale) * noise_cfg
    return noise_cfg


def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image

class StableDiffusionImage2PanoPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. If you set multiple ControlNets
            as a list, the outputs from each ControlNet are added together to create one combined additional
            conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae, controlnet, and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.controlnet]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            # the safety checker can offload the vae again
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # control net hook has be manually offloaded as it alternates with unet
        cpu_offload_with_hook(self.controlnet, device)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        max_embeddings_multiples=3,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
        """
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
        if prompt_embeds is None or negative_prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
                if do_classifier_free_guidance and negative_prompt_embeds is None:
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, self.tokenizer)

            prompt_embeds1, negative_prompt_embeds1 = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,
                uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                max_embeddings_multiples=max_embeddings_multiples,
            )
            if prompt_embeds is None:
                prompt_embeds = prompt_embeds1
            if negative_prompt_embeds is None:
                negative_prompt_embeds = negative_prompt_embeds1

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            bs_embed, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        image,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(image, prompt, prompt_embeds)
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(image, list):
                raise TypeError("For multiple controlnets: `image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in image):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif len(image) != len(self.controlnet.nets):
                raise ValueError(
                    "For multiple controlnets: `image` must have the same length as the number of controlnets."
                )

            for image_ in image:
                self.check_image(image_, prompt, prompt_embeds)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)

        if not image_is_pil and not image_is_tensor and not image_is_pil_list and not image_is_tensor_list:
            raise TypeError(
                "image must be passed and be one of PIL image, torch tensor, list of PIL images, or list of torch tensors"
            )

        if image_is_pil:
            image_batch_size = 1
        elif image_is_tensor:
            image_batch_size = image.shape[0]
        elif image_is_pil_list:
            image_batch_size = len(image)
        elif image_is_tensor_list:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.prepare_latents
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def _default_height_width(self, height, width, image):
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    # override DiffusionPipeline
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = False,
        variant: Optional[str] = None,
    ):
        if isinstance(self.controlnet, ControlNetModel):
            super().save_pretrained(save_directory, safe_serialization, variant)
        else:
            raise NotImplementedError("Currently, the `save_pretrained()` is not implemented for Multi-ControlNet.")

    def denoise_latents(self, latents, t, prompt_embeds, control_image, controlnet_conditioning_scale, guess_mode, cross_attention_kwargs, do_classifier_free_guidance, guidance_scale, extra_step_kwargs, views_scheduler_status):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        self.scheduler.__dict__.update(views_scheduler_status[0])
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # controlnet(s) inference
        if guess_mode and do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            controlnet_latent_model_input = latents
            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
        else:
            controlnet_latent_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            controlnet_latent_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=control_image,
            conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )

        if guess_mode and do_classifier_free_guidance:
            # Infered ControlNet only for the conditional batch.
            # To apply the output of ControlNet to both the unconditional and conditional batches,
            # add 0 to the unconditional batch to keep it unchanged.
            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        return latents

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def get_blocks(self, latents, control_image, tile_latent_min_size, overlap_size):
        rows_latents = []
        rows_control_images = []
        for i in range(0, latents.shape[2] - overlap_size, overlap_size):
            row_latents = []
            row_control_images = []
            for j in range(0, latents.shape[3] - overlap_size, overlap_size):
                latents_input = latents[:, :, i: i + tile_latent_min_size, j: j + tile_latent_min_size]
                control_image_input = control_image[:, :,
                                      self.vae_scale_factor * i: self.vae_scale_factor * (i + tile_latent_min_size),
                                      self.vae_scale_factor * j: self.vae_scale_factor * (j + tile_latent_min_size)]
                row_latents.append(latents_input)
                row_control_images.append(control_image_input)
            rows_latents.append(row_latents)
            rows_control_images.append(row_control_images)
        return rows_latents, rows_control_images

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        control_image: Union[
            torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        mask: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,
                    `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list. Note that by default, we use a smaller conditioning scale for inpainting
                than for [`~StableDiffusionControlNetPipeline.__call__`].
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        def controlnet_forward(
                self,
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor,
                controlnet_cond: torch.FloatTensor,
                conditioning_scale: float = 1.0,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                guess_mode: bool = False,
                return_dict: bool = True,
                mask: Optional[torch.FloatTensor] = None,
        ) -> Union[ControlNetOutput, Tuple]:
            """
            The [`ControlNetModel`] forward method.

            Args:
                sample (`torch.FloatTensor`):
                    The noisy input tensor.
                timestep (`Union[torch.Tensor, float, int]`):
                    The number of timesteps to denoise an input.
                encoder_hidden_states (`torch.Tensor`):
                    The encoder hidden states.
                controlnet_cond (`torch.FloatTensor`):
                    The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
                conditioning_scale (`float`, defaults to `1.0`):
                    The scale factor for ControlNet outputs.
                class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                    Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
                timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                added_cond_kwargs (`dict`):
                    Additional conditions for the Stable Diffusion XL UNet.
                cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                    A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
                guess_mode (`bool`, defaults to `False`):
                    In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                    you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
                return_dict (`bool`, defaults to `True`):
                    Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

            Returns:
                [`~models.controlnet.ControlNetOutput`] **or** `tuple`:
                    If `return_dict` is `True`, a [`~models.controlnet.ControlNetOutput`] is returned, otherwise a tuple is
                    returned where the first element is the sample tensor.
            """
            # check channel order
            channel_order = self.config.controlnet_conditioning_channel_order

            if channel_order == "rgb":
                # in rgb order by default
                ...
            elif channel_order == "bgr":
                controlnet_cond = torch.flip(controlnet_cond, dims=[1])
            else:
                raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # 1. time
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])

            t_emb = self.time_proj(timesteps)

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=sample.dtype)

            emb = self.time_embedding(t_emb, timestep_cond)
            aug_emb = None

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")

                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)

                class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                emb = emb + class_emb

            if self.config.addition_embed_type is not None:
                if self.config.addition_embed_type == "text":
                    aug_emb = self.add_embedding(encoder_hidden_states)

                elif self.config.addition_embed_type == "text_time":
                    if "text_embeds" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                        )
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    if "time_ids" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                        )
                    time_ids = added_cond_kwargs.get("time_ids")
                    time_embeds = self.add_time_proj(time_ids.flatten())
                    time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                    add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                    add_embeds = add_embeds.to(emb.dtype)
                    aug_emb = self.add_embedding(add_embeds)

            emb = emb + aug_emb if aug_emb is not None else emb

            # 2. pre-process
            sample = self.conv_in(sample)

            controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)

            if mask is not None:
                sample = (1 - mask.to(sample.dtype)) * sample + mask.to(sample.dtype) * controlnet_cond
            else:
                sample = sample + controlnet_cond

            # 3. down
            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                down_block_res_samples += res_samples

            # 4. mid
            if self.mid_block is not None:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

            # 5. Control net blocks

            controlnet_down_block_res_samples = ()

            for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
                down_block_res_sample = controlnet_block(down_block_res_sample)
                controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = controlnet_down_block_res_samples

            mid_block_res_sample = self.controlnet_mid_block(sample)

            # 6. scaling
            if guess_mode and not self.config.global_pool_conditions:
                scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0

                scales = scales * conditioning_scale
                down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
                mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
            else:
                down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
                mid_block_res_sample = mid_block_res_sample * conditioning_scale

            if self.config.global_pool_conditions:
                down_block_res_samples = [
                    torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
                ]
                mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

            if not return_dict:
                return (down_block_res_samples, mid_block_res_sample)

            return ControlNetOutput(
                down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
            )
        self.controlnet.forward = controlnet_forward.__get__(self.controlnet, ControlNetModel)

        def tiled_decode(
            self,
            z: torch.FloatTensor,
            return_dict: bool = True
        ) -> Union[DecoderOutput, torch.FloatTensor]:
            r"""Decode a batch of images using a tiled decoder.

            Args:
            When this option is enabled, the VAE will split the input tensor into tiles to compute decoding in several
            steps. This is useful to keep memory use constant regardless of image size. The end result of tiled
            decoding is: different from non-tiled decoding due to each tile using a different decoder.
            To avoid tiling artifacts, the tiles overlap and are blended together to form a smooth output.
            You may still see tile-sized changes in the look of the output, but they should be much less noticeable.
                z (`torch.FloatTensor`): Input batch of latent vectors. return_dict (`bool`, *optional*, defaults to
                `True`):
                    Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            """
            _tile_overlap_factor = 1 - self.tile_overlap_factor
            overlap_size = int(self.tile_latent_min_size
                               * _tile_overlap_factor)
            blend_extent = int(self.tile_sample_min_size
                               * self.tile_overlap_factor)
            row_limit = self.tile_sample_min_size - blend_extent
            w = z.shape[3]
            z = torch.cat([z, z[:, :, :, :w // 4]], dim=-1)
            # Split z into overlapping 64x64 tiles and decode them separately.
            # The tiles have an overlap to avoid seams between tiles.

            rows = []
            for i in range(0, z.shape[2], overlap_size):
                row = []
                tile = z[:, :, i:i + self.tile_latent_min_size, :]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                vae_scale_factor = decoded.shape[-1] // tile.shape[-1]
                row.append(decoded)
                rows.append(row)
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(
                        self.blend_h(
                            tile[:, :, :row_limit, w * vae_scale_factor:],
                            tile[:, :, :row_limit, :w * vae_scale_factor],
                            tile.shape[-1] - w * vae_scale_factor))
                result_rows.append(torch.cat(result_row, dim=3))

            dec = torch.cat(result_rows, dim=2)
            if not return_dict:
                return (dec, )

            return DecoderOutput(sample=dec)

        self.vae.tiled_decode = tiled_decode.__get__(self.vae, AutoencoderKL)

        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)
        self.blend_extend = width // self.vae_scale_factor // 32

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            control_image,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        self.controlnet.to(device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        # 4. Prepare image, and controlnet_conditioning_image
        image = prepare_image(image)

        # 5. Prepare image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)

            control_image = control_images
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )
        if mask is not None:
            mask = torch.cat([mask] * batch_size, dim=0)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        views_scheduler_status = [copy.deepcopy(self.scheduler.__dict__)]
        # value = torch.zeros_like(latents)
        latents = torch.cat([latents, latents[:, :, :, :self.blend_extend]], dim=-1)
        control_image = torch.cat([control_image, control_image[:, :, :, :self.blend_extend * self.vae_scale_factor]], dim=-1)
        if mask is not None:
            mask = torch.cat([mask] * batch_size, dim=0)
            mask = torch.cat([mask, mask[:, :, :, :self.blend_extend]], dim=-1)


        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if mask is not None:
                    mask_input = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
                else:
                    mask_input = None

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    controlnet_latent_model_input = latents
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    controlnet_latent_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    controlnet_latent_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=controlnet_conditioning_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                    mask=mask_input,
                )

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                # latents = value + 0.0
                latents = self.blend_h(latents, latents, self.blend_extend)
            latents = self.blend_h(latents, latents, self.blend_extend)
            latents = latents[:, :, :, :width // self.vae_scale_factor]

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
