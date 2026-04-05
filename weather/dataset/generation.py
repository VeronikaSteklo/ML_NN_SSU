import os

from diffusers import StableDiffusionPipeline
import torch
from diffusers.utils import logging

logging.set_verbosity_info()


def generate_ai_weather(save_dir, class_names, n_per_class=10):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("mps")

    for cls in class_names:
        print("Generating " + cls)
        target_path = os.path.join(save_dir, cls)
        os.makedirs(target_path, exist_ok=True)

        prompt = f"Professional realistic photo of {cls} weather, high resolution, detailed"

        for i in range(n_per_class):
            image = pipe(prompt, num_inference_steps=20).images[0]
            image.save(os.path.join(target_path, f"ai_{i}.png"))
