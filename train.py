import torch.nn as nn
import torch.optim as optim
from GameActionModel import GameActionModel
from setting import Task, NN_FILE_NAME, TRAINING_DATA_NAME
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import csv


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


dataset = GameActionDataset(TRAINING_DATA_NAME)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


model = GameActionModel(num_tasks=len(Task))
model.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):  # number of epochs
    for images, task_ids, labels in dataloader:
        output = model(images, task_ids)  # [B, 3]
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), NN_FILE_NAME)
