import os
import csv
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch
from tqdm import tqdm

# === settings ===
AUG_PER_IMAGE = 5

# === Saved data paths ===
directory = "train_data"
image_path = os.path.join(directory, "images")
data_path = os.path.join(directory, "data")
csv_name = os.path.join(data_path, "clean_data.csv")

# === destination path ===
dest_directory = "augmented_data"
dest_image_path = os.path.join(dest_directory, "images")
dest_data_path = os.path.join(dest_directory, "data")
dest_csv_name = os.path.join(dest_data_path, "augmented_data.csv")

# === Create folders if needed ===
os.makedirs(dest_image_path, exist_ok=True)
os.makedirs(dest_data_path, exist_ok=True)

file_exists = os.path.exists(dest_csv_name)

if not file_exists:
    with open(dest_csv_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header only once when file is new
        writer.writerow(["image_path", "task", "x_norm",
                        "y_norm", "done", "confidence"])

# === Define Transformations ===
augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Unnormalize values (mean & std from ImageNet)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# === Read Original CSV and Augment ===
with open(csv_name, "r") as infile, open(dest_csv_name, "w", newline='') as outfile:
    reader = csv.reader(infile)
    next(reader)

    writer = csv.writer(outfile)

    writer.writerow(["image_path", "task", "x_norm",
                    "y_norm", "done", "confidence"])

    augmented_count = 0
    write_count = 0
    for row in tqdm(reader, desc="Augmenting and updating CSV"):
        if len(row) < 6:
            continue  # Skip malformed rows

        img_path, task_id, x, y, done, confidence = row

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[!] Error reading {img_path}: {e}")
            continue

        augmented_count += 1
        for i in range(AUG_PER_IMAGE):
            
            to_tensor = transforms.ToTensor()
            aug_tensor = to_tensor(image)  # now aug_tensor is a torch.Tensor
            unnormalized = aug_tensor * std + mean
            unnormalized = torch.clamp(unnormalized, 0, 1)

            new_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_aug{i}.png"
            new_path = os.path.join(dest_image_path, new_filename)
            save_image(unnormalized, new_path)

            writer.writerow([new_path, task_id, x, y, done, confidence])

            write_count += 1

    print(f"{augmented_count} data(s) augmented.")
    print(f"{write_count} data(s) wrote.")
