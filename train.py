import torch.nn as nn
import torch.optim as optim
from GameActionModel import GameActionModel
from setting import Task, NN_FILE_NAME, SCRIPT_DIR, TRAINING_DATA_PATH
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from resize_image import ResizeImage
import torch
import csv
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
import os

console = Console()

# === Detect GPU ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(f"[bold yellow]Using device:[/] {device}")

class GameActionDataset(Dataset):
    def __init__(self, label_file):
        self.data = []
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append(row)

        self.transform = ResizeImage(size=224)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, task_id, x, y, done, confidence = self.data[idx]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        task_id = torch.tensor(int(task_id), dtype=torch.long)
        label = torch.tensor(
            [float(x), float(y), float(done), float(confidence)],
            dtype=torch.float
        )

        return image, task_id, label


# === Paths ===
augmented_data_path = os.path.join(SCRIPT_DIR, "augmented_data")
augmented_data = os.path.join(augmented_data_path, f"augmented_data.csv")
training_data = os.path.join(TRAINING_DATA_PATH, "data2025-06-21.csv")
data_sets = [augmented_data, training_data]

# === Model & Optimizer ===
model = GameActionModel(num_tasks=len(Task)).to(device)
model.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Training ===
for data in data_sets:
    print(f"Training model with data: {data}")
    dataset = GameActionDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(10):  # Number of epochs per dataset
        running_loss = 0.0

        with Progress(
            TextColumn("[bold blue]Epoch {}/10".format(epoch + 1)),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Training", total=len(dataloader))

            for images, task_ids, labels in dataloader:
                # Move to GPU
                images = images.to(device)
                task_ids = task_ids.to(device)
                labels = labels.to(device)

                output = model(images, task_ids)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress.advance(task)

        avg_loss = running_loss / len(dataloader)
        console.print(
            f"[green]Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}[/green]"
        )

# === Save Model ===
torch.save(model.state_dict(), NN_FILE_NAME)
console.print(f"[bold green]Model saved to {NN_FILE_NAME}[/bold green]")