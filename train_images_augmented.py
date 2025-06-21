import os
import csv
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch
from tqdm import tqdm
from setting import TRAINING_DATA_NAME, TRAINING_IMAGE_PATH, AUGMENTED_IMAGE_PATH, AUGMENTED_DATA_NAME

# === Paths ===
CSV_FILE = TRAINING_DATA_NAME
SOURCE_DIR = TRAINING_IMAGE_PATH
DEST_DIR = AUGMENTED_IMAGE_PATH
OUTPUT_CSV = AUGMENTED_DATA_NAME

# === Create folders if needed ===
os.makedirs(DEST_DIR, exist_ok=True)

# === Define Transformations ===
augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Read Original CSV and Augment ===
with open(CSV_FILE, "r") as infile, open(OUTPUT_CSV, "w", newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in tqdm(reader, desc="Augmenting and updating CSV"):
        if len(row) < 6:
            continue  # skip malformed rows

        img_path, task_id, x, y, done, confidence = row
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            continue

        for i in range(AUG_PER_IMAGE):
            aug_tensor = augmentation(image)
            # unnormalize for saving
            unnormalized = aug_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

            new_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_aug{i}.png"
            new_path = os.path.join(DEST_DIR, new_filename)

            save_image(unnormalized, new_path)

            # Write new row with same label values but new image path
            writer.writerow([new_path, task_id, x, y, done, confidence])
