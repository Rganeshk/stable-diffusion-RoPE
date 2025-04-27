import torch
from torchvision import transforms
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel

# === Config ===
device = "cuda" if torch.cuda.is_available() else "cpu"
image_folder = "./generated_images_alibi"  # your folder
prompts = [
    "three cats sitting on a fence",
    "a group of seven apples on a table",
    "five colorful balloons in the sky",
    "two cars parked near a tree",
    "four birds flying over the ocean"
]

# === Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Helper function
def compute_clip_score(image_path, prompt):
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    clip_score = (image_embeds * text_embeds).sum(dim=-1)  # cosine similarity
    return clip_score.item()

# === Main
all_clip_scores = []
for idx, prompt in enumerate(prompts):
    #idx += 1  # Adjust index for file naming
    image_path = os.path.join(image_folder, f"image_{idx}.png")
    if not os.path.exists(image_path):
        print(f"[WARN] Image not found: {image_path}")
        continue

    score = compute_clip_score(image_path, prompt)
    all_clip_scores.append(score)
    print(f"[INFO] Prompt: {prompt}")
    print(f"      CLIP Score: {score:.4f}")

# === Average CLIP Score
avg_clip_score = sum(all_clip_scores) / len(all_clip_scores)
print(f"\n=== Final Result ===")
print(f"Average CLIP Score over {len(all_clip_scores)} prompts: {avg_clip_score:.4f}")
