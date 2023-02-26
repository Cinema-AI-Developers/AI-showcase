from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from tokenizer import tokenizer, text_encoder

model_id = "stabilityai/stable-diffusion-2-1-base"


scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                               scheduler=scheduler,
                                               text_encoder=text_encoder,
                                               tokenizer=tokenizer,
                                               revision="fp16",
                                               torch_dtype=torch.float16,
                                               safety_checker=None)
pipe = pipe.to("cuda")