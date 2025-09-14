import os, shutil, random

# Paths
base_dir = "/Users/berry/Desktop/firedetect/fire_dataset/fire_dataset"
no_fire_src = "/Users/berry/Desktop/firedetect/fire_dataset/non_fire_images"

train_dir = os.path.join(base_dir, "train/no_fire")
val_dir   = os.path.join(base_dir, "val/no_fire")
test_dir  = os.path.join(base_dir, "test/no_fire")

# Create folders
for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

# Get all images
all_imgs = [f for f in os.listdir(no_fire_src) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
print(f"🔎 Found {len(all_imgs)} no_fire images in source folder.")

random.shuffle(all_imgs)

# Split 70/20/10
n = len(all_imgs)
train_split = int(0.7 * n)
val_split   = int(0.9 * n)

for i, img in enumerate(all_imgs):
    src = os.path.join(no_fire_src, img)
    if i < train_split:
        dst = os.path.join(train_dir, img)
    elif i < val_split:
        dst = os.path.join(val_dir, img)
    else:
        dst = os.path.join(test_dir, img)
    
    shutil.copy2(src, dst)  # copy2 keeps metadata, overwrites if needed

print("✅ Images copied into train/val/test folders.")