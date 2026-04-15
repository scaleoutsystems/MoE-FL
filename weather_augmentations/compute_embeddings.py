import numpy as np
import torch
import clip
from torch.utils.data import DataLoader
from utils.data_utils import ImageNetSubset, WeatherImageNetSubset
from tqdm import tqdm

#Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model.eval()

#Load datasets

orig_dataset = ImageNetSubset(
    root="data/imnet_subset",
    split="train",
    transform=preprocess
)

aug_dataset = WeatherImageNetSubset(
    root="data/imnet_subset_weather_aug",
    split="train",
    transform=preprocess
)

#Alignment check

assert len(orig_dataset) == len(aug_dataset), \
    f"Size mismatch: {len(orig_dataset)} orig vs {len(aug_dataset)} aug"

assert orig_dataset.targets == aug_dataset.targets, \
    "Label mismatch between original and augmented datasets"

print(f"Alignment verified - {len(orig_dataset)} image pairs")
print(f"Weather distribution: { {w: aug_dataset.weathers.count(w) for w in aug_dataset.weather_types} }")

#Embedding extraction
def extract_embeddings(dataset, model, device, batch_size=64):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    embeddings = []
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for images, _ in tqdm(loader, desc="Extracting embeddings"):
            emb = model.encode_image(images.to(device))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings)

print("\nExtracting original embeddings...")
orig_emb = extract_embeddings(orig_dataset, model, device)

print("\nExtracting augmented embeddings...")
aug_emb = extract_embeddings(aug_dataset, model, device)

# ── Save ──────────────────────────────────────────────────────────────────────

np.save("embeddings_original.npy", orig_emb)
np.save("embeddings_augmented.npy", aug_emb)
np.save("class_ids.npy", np.array(orig_dataset.targets))
np.save("weather_labels.npy", np.array(aug_dataset.weathers))

print(f"\nDone.")
print(f"embeddings_original.npy : {orig_emb.shape}")
print(f"embeddings_augmented.npy : {aug_emb.shape}")
print(f"class_ids.npy : {len(orig_dataset.targets)}")
print(f"weather_labels.npy : {len(aug_dataset.weathers)}")