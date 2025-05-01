import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
import wandb
wandb.login(key="c85018ad904289a078e360569b438647e38c3951")
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

# === Config ===
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
epochs = 25
lr = 1e-4
max_length = 77
save_dir = "./checkpoints_alibi"
os.makedirs(save_dir, exist_ok=True)
save_every_n_steps = 500
subset_size = 50000
margin = 0.5  # for contrastive loss

# === WandB Init ===
wandb.init(
    project="clip-finetune-alibi",
    name="alibi_contrastive_loss_run",
    config={
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "margin": margin,
        "subset_size": subset_size,
    }
)

# === Dataset ===
class CocoCountingDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", tokenizer=None, max_length=77):
        self.dataset = load_dataset("conceptual_captions", split=split)
        self.dataset = self.dataset.select(range(subset_size))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        caption = self.dataset[idx]['caption'].lower()
        label = int(any(word in caption for word in self.number_words))
        if label == 0:
            caption = "one object."
        encoding = self.tokenizer(caption, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return input_ids, attention_mask, label

# === Model ===
model = FrozenCLIPEmbedder(
    version="openai/clip-vit-large-patch14",
    device=device,
    max_length=max_length,
    rope=False,
    alibi=True
)
model.to(device)

# === Unfreeze transformer parameters for finetuning
for param in model.transformer.parameters():
    param.requires_grad = True

# === Optimizer ===
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

# === Dataloader ===
print("[INFO] Preparing dataloader...")
dataset = CocoCountingDataset(split="train", tokenizer=model.tokenizer, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# === Training ===
model.train()
global_step = 0
for epoch in range(epochs):
    total_loss = 0

    print(f"[INFO] Starting epoch {epoch+1}/{epochs}...")

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(tqdm(dataloader)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Contrastive loss for counting
        pos_mask = labels == 1
        neg_mask = labels == 0

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_embeds = embeddings[pos_mask]
            neg_embeds = embeddings[neg_mask]
            anchor = pos_embeds
            positive = pos_embeds
            negative = neg_embeds[torch.randint(0, neg_embeds.size(0), (pos_embeds.size(0),)).to(device)]

            pos_sim = F.cosine_similarity(anchor, positive)
            neg_sim = F.cosine_similarity(anchor, negative)

            contrastive_loss = F.relu(margin + neg_sim - pos_sim).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss = contrastive_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        # Log loss and margin satisfaction to WandB
        with torch.no_grad():
            margin_violations = (margin + neg_sim - pos_sim) > 0
            margin_satisfaction_rate = 1.0 - margin_violations.float().mean().item()

        wandb.log({
            "Loss": loss.item(),
            "Margin Satisfaction": margin_satisfaction_rate
        }, step=global_step)

        if global_step % 100 == 0:
            print(f"[DEBUG] Step {global_step}: Loss={loss.item():.4f}")

        if global_step % save_every_n_steps == 0:
            checkpoint_path = os.path.join(save_dir, f"clip_alibi_step{global_step}.pth")
            torch.save(model.transformer.state_dict(), checkpoint_path)
            print(f"[Checkpoint] Saved at step {global_step}")

    print(f"Epoch {epoch+1}: Average Loss={total_loss/len(dataloader):.4f}")
    checkpoint_path = os.path.join(save_dir, f"clip_alibi_epoch{epoch+1}.pth")
    torch.save(model.transformer.state_dict(), checkpoint_path)
    print(f"[Checkpoint] Saved after epoch {epoch+1}")

# === Final Save ===
torch.save(model.transformer.state_dict(), "./clip_alibi_final.pth")
print("[Final Save] Fine-tuned ALiBi text encoder saved!")

wandb.finish()