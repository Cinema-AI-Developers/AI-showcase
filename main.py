import os

import gradio as gr
import torch

from pipe import pipe

model_id = "stabilityai/stable-diffusion-2-1-base"


def interference(prompt="", num_inference_steps=50, guidance_scale=7.5, save_img=False, show_img=False):
    def prompt_builder(prompt):
        return f'Real film cover with no text for online cinema for film "{prompt}" in the style of <midjourney>'

    negative_prompt = "in the style of <wrong>"
    pos_prompt = prompt_builder(prompt)

    with torch.autocast("cuda"):
        image = pipe(pos_prompt,
                     negative_prompt=negative_prompt,
                     num_inference_steps=num_inference_steps,
                     guidance_scale=guidance_scale).images[0]
    if show_img:
        print(prompt)
        display(image)

    if save_img:
        if not os.path.exists("outputs"):
            os.mkdir("outputs")

        prompt_folder = prompt.replace("/", "_")[0:128]

        if not os.path.exists(os.path.join("outputs", prompt_folder)):
            os.mkdir(os.path.join("outputs", prompt_folder))

        image.save(os.path.join("outputs", prompt_folder, f"{prompt}.png"))
    return image


app = gr.Interface(
    fn=interference,
    inputs=gr.inputs.Textbox(label="Input Text"),
    outputs=gr.outputs.Image(type="pil", label="Output Image")
)

app.launch(debug=True)
