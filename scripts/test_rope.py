# test_rope.py

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# === Config ===
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "./generated_images"
os.makedirs(output_dir, exist_ok=True)

# === Load Stable Diffusion pipeline ===
print("[INFO] Loading Stable Diffusion pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# === Load your fine-tuned LoRA weights ===
print("[INFO] Loading LoRA fine-tuned weights...")
lora_checkpoint_path = "./checkpoints/clip_rope_lora_epoch8.pth"  # Change if needed
lora_state_dict = torch.load(lora_checkpoint_path, map_location=device)

missing, unexpected = pipe.text_encoder.load_state_dict(lora_state_dict, strict=False)
print(f"[INFO] LoRA Loaded! Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

# === Prepare test prompts ===
prompts = [
    "three cats sitting on a fence",
    "a group of seven apples on a table",
    "five colorful balloons in the sky",
    "two cars parked near a tree",
    "four birds flying over the ocean"
]

# === Disable safety checker if needed
pipe.safety_checker = None  # (optional)

# === Generate images ===
print("[INFO] Generating images with fine-tuned model...")

for idx, prompt in enumerate(prompts):
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu"):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    
    save_path = os.path.join(output_dir, f"image_{idx+1}.png")
    image.save(save_path)
    print(f"[INFO] Saved: {save_path}  (Prompt: '{prompt}')")

print("[INFO] Generation complete!")
