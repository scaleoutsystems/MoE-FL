import numpy as np
import torch
import clip
from utils.data_utils import ImageNetSubset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model.eval()

# Load dataset just to get the remapped class names
dataset = ImageNetSubset(root="data/imnet_subset", split="train", transform=None)
# dataset.classes is a dict {0: 'name', 1: 'name', ...} sorted by remapped label
prompts = [f"a photo of a {dataset.classes[i]}" for i in range(dataset.num_classes)]

with torch.no_grad():
    text_tokens = clip.tokenize(prompts).to(device)
    text_emb = model.encode_text(text_tokens)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

np.save("embeddings_labels.npy", text_emb.cpu().numpy())
print(f"embeddings_labels.npy : {text_emb.shape}")  # (100, D)