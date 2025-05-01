import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from diffusers import StableDiffusionPipeline
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
import os
from PIL import Image

# === Config ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_ckpt = "./checkpoints_alibi/clip_alibi_epoch17.pth"  # <- your fine-tuned checkpoint
output_dir = "./generated_images_alibi"
os.makedirs(output_dir, exist_ok=True)

prompts = [
    "three cats sitting on a fence",
    "a group of seven apples on a table",
    "five colorful balloons in the sky",
    "two cars parked near a tree",
    "four birds flying over the ocean"
]

# === Load Stable Diffusion Pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    revision="fp16",
    safety_checker=None,  # optionally disable NSFW filter
).to(device)

# === Load Fine-tuned Text Encoder
custom_text_encoder = FrozenCLIPEmbedder(
    version="openai/clip-vit-large-patch14",
    device=device,
    rope=False,
    alibi=True
)
custom_text_encoder.transformer.load_state_dict(torch.load(model_ckpt, map_location=device))
custom_text_encoder.transformer = custom_text_encoder.transformer.eval().to(device).half()

# === Replace Text Encoder inside the Pipeline
pipe.text_encoder = custom_text_encoder.transformer
pipe.tokenizer = custom_text_encoder.tokenizer

# === Move entire pipeline to device
pipe = pipe.to(device)

# === Generate images
pipe.enable_attention_slicing()  # to save memory

for idx, prompt in enumerate(prompts):
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(f"{output_dir}/image_{idx}.png")
    print(f"Saved image for prompt [{prompt}] at {output_dir}/image_{idx}.png")

print("✅ Done! Images generated.")
