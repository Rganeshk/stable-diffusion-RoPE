import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from collections import Counter
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# === Configuration ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# Folder containing generated images
image_folder = "./generated_images_alibi"

# Prompts corresponding to images
prompts = [
    "three cats sitting on a fence",
    "a group of seven apples on a table",
    "five colorful balloons in the sky",
    "two cars parked near a tree",
    "four birds flying over the ocean"
]

# Expected counts per prompt
expected_counts = [3, 7, 5, 2, 4]

# === Load Detection Model ===
model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.eval().to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# === Helper Functions ===
def detect_objects(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).to(device)
    with torch.no_grad():
        outputs = model([img_tensor])[0]
    # Filter out low confidence detections
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    boxes = outputs['boxes'].cpu().numpy()
    keep = scores > 0.5
    labels = labels[keep]
    return labels

# === Main Evaluation ===
predicted_counts = []
for idx, prompt in enumerate(prompts):
    image_path = os.path.join(image_folder, f"image_{idx}.png")
    if not os.path.exists(image_path):
        print(f"[WARN] Image not found: {image_path}")
        continue

    labels = detect_objects(image_path)
    num_objects_detected = len(labels)
    predicted_counts.append(num_objects_detected)

    print(f"[INFO] {prompt}")
    print(f"Expected count: {expected_counts[idx]}, Detected count: {num_objects_detected}")

# === Calculate Metrics ===
# Treat as binary: "correct" if count matches within +/- 1 tolerance
correct = []
for pred, true in zip(predicted_counts, expected_counts):
    if abs(pred - true) <= 1:  # allow small tolerance
        correct.append(1)
    else:
        correct.append(0)

precision, recall, f1, _ = precision_recall_fscore_support(
    [1]*len(correct),  # ground truth all ones
    correct,           # predictions
    average='binary'
)

print("\n=== Evaluation Results ===")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
