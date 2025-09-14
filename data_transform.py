import os
import shutil
import random
import yaml  # pip install pyyaml

# Path to your downloaded PNG files
dataset_dir = "./fire_images"  # all PNGs here
output_dir = "./fire_dataset"

os.makedirs(output_dir, exist_ok=True)
splits = ["train", "val", "test"]

# Create folders
for split in splits:
    for cls in ["fire", "no_fire"]:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# List images
all_images = [f for f in os.listdir(dataset_dir) if f.endswith(".png")]
random.shuffle(all_images)

# Split ratios
train_ratio, val_ratio = 0.7, 0.15
n_total = len(all_images)
n_train = int(train_ratio * n_total)
n_val = int(val_ratio * n_total)

# Move files
for i, img in enumerate(all_images):
    if i < n_train:
        split = "train"
    elif i < n_train + n_val:
        split = "val"
    else:
        split = "test"

    # Simple label assignment: if filename contains "fire" -> fire, else no_fire
    label = "fire" if "fire" in img.lower() else "no_fire"

    shutil.copy(os.path.join(dataset_dir, img),
                os.path.join(output_dir, split, label, img))

# -------------------
# Create YAML metadata
# -------------------
metadata = {}
for split in splits:
    metadata[split] = {}
    for cls in ["fire", "no_fire"]:
        cls_dir = os.path.join(output_dir, split, cls)
        metadata[split][cls] = os.listdir(cls_dir)

with open(os.path.join(output_dir, "metadata.yml"), "w") as f:
    yaml.dump(metadata, f)

print("✅ Dataset split and YAML metadata created!")