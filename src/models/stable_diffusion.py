from typing import List
from PIL.Image import Image
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("mps")

def stable_diffustion(text: str) -> List[Image]:
  result = pipe(prompt=text, num_inference_steps=25).images
  print(result)
  return result

def ui():
  return gr.Interface(
    stable_diffustion,
    inputs=["textbox"],
    outputs=[gr.Gallery()],
  )

if __name__ == "__main__":
  ui().launch()
