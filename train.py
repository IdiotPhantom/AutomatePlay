import torch.nn as nn
import torch.optim as optim
from GameActionModel import GameActionModel
from setting import Task, NN_FILE_NAME, SCRIPT_DIR, TRAINING_DATA_PATH
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import csv
from tqdm import tqdm
import os


class GameActionDataset(Dataset):
    def __init__(self, label_file):
        self.data = []
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append(row)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Expecting: path, task_id, x, y, done, confidence
        path, task_id, x, y, done, confidence = self.data[idx]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        task_id = torch.tensor(int(task_id), dtype=torch.long)
        label = torch.tensor(
            [float(x), float(y), float(done), float(confidence)],
            dtype=torch.float
        )

        return image, task_id, label


# Format it as a date string (e.g., YYYY-MM-DD)
augmented_data_path = os.path.join(SCRIPT_DIR, "augmented_data")
augmented_data = os.path.join(augmented_data_path, f"data.csv")

training_data = os.path.join(TRAINING_DATA_PATH,"data2025-06-21.csv")

dataset = GameActionDataset(training_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


model = GameActionModel(num_tasks=len(Task))
model.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):  # number of epochs
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")

    for images, task_ids, labels in progress_bar:
        output = model(images, task_ids)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    print(
        f"Epoch {epoch+1} completed. Avg Loss: {running_loss / len(dataloader):.4f}")


torch.save(model.state_dict(), NN_FILE_NAME)
