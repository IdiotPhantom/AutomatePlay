# === learn.py (fixed version) ===
import os
import csv
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import keyboard
import torch.nn.functional as F
from collections import Counter
from PIL import Image

from GameActionModel import GameActionModel
from resize_image import ResizeImage
from setting import Task, SCREEN_HEIGHT, SCREEN_WIDTH
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console


def is_within_radius(x_pred, y_pred, x_target, y_target, radius=30):
    return (abs(x_pred - x_target) <= radius) and (abs(y_pred - y_target) <= radius)


class Learn:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = GameActionModel(num_tasks=len(Task)).to(self.device)
        self.model.train()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.feedback_counter = 0
        self.acc_tap = deque(maxlen=100)
        self.acc_skip = deque(maxlen=100)
        self.acc_done = deque(maxlen=100)
        self.correct_buffer = deque(maxlen=100)

    def decide_action(self, image_path, task: Task, return_raw_output=False):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert("L")
        except Exception as e:
            print(f"[!] Error loading image: {image_path} -- {e}")
            raise

        tensor_image, scale, padding = ResizeImage.preprocess_image(image)
        task_tensor = torch.tensor(
            [task.value], dtype=torch.long).to(self.device)
        tensor_image = tensor_image.to(self.device)

        self.model.eval()
        with torch.no_grad():
            x_y, done_score, confidence = self.model(tensor_image, task_tensor)

        x_norm, y_norm = x_y[0]
        x, y = ResizeImage.unpad_and_rescale_coords(
            x_norm.item(), y_norm.item(), scale, padding,
            orig_w=SCREEN_WIDTH, orig_h=SCREEN_HEIGHT
        )

        action = {}
        if done_score.item() > 0.7:
            action["type"] = "done"
        elif confidence.item() < 0.5:
            action["type"] = "skip"
        else:
            action.update({
                "type": "tap",
                "x": x,
                "y": y,
                "x_norm": x_norm.item(),
                "y_norm": y_norm.item(),
                "confidence": confidence.item()
            })

        if return_raw_output:
            action["raw_output"] = (x_y, done_score, confidence)

        return action

    def learn(self):
        success_count = 0
        failed_count = 0
        total_loss = 0
        count = 0

        self.acc_done.extend([0, 1])
        self.acc_skip.extend([0, 1])
        self.acc_tap.extend([0, 1])

        with Live(console=Console(), refresh_per_second=4) as live:
            while True:
                if keyboard.is_pressed('esc'):
                    print("ESC pressed, exiting loop.")
                    break

                count += 1

                accuracy = self.avg(self.correct_buffer)
                acc_done = self.avg(self.acc_done)
                acc_skip = self.avg(self.acc_skip)
                acc_tap = self.avg(self.acc_tap)

                lr: float = 1e-3  # default value

                if accuracy < 0.1:
                    lr = 3e-3
                elif accuracy < 0.3:
                    lr = 1e-3
                elif accuracy < 0.6:
                    lr = 1e-4
                elif accuracy > 0.6:
                    lr = 1e-5
                elif accuracy > 0.9:
                    break

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                rows: list = []

                choice = min(acc_tap, acc_done, acc_skip, 0.5)
                if choice == acc_tap:
                    rows = tap_row
                elif choice == acc_done:
                    rows = done_row
                elif choice == acc_skip:
                    rows = skip_row

                if random.random() > 0.6:
                    rows = all_rows

                data = random.choice(rows)
                image = data['image_path']
                task = Task(int(data['task']))
                correct_x = float(data['x_norm']) * SCREEN_WIDTH
                correct_y = float(data['y_norm']) * SCREEN_HEIGHT
                correct_done = float(data['done'])
                correct_confidence = float(data['confidence'])

                action = self.decide_action(image, task)
                is_current_correct = False

                if action["type"] == "done":
                    is_current_correct = correct_done >= 0.9
                elif action["type"] == "tap":
                    if correct_confidence >= 0.9:
                        is_current_correct = is_within_radius(
                            action['x'], action['y'], correct_x, correct_y)
                    else:
                        is_current_correct = False
                elif action["type"] == "skip":
                    is_current_correct = correct_confidence <= 0.1

                self.correct_buffer.append(is_current_correct)
                getattr(self, f"acc_{action['type']}").append(
                    is_current_correct)

                if is_current_correct:
                    success_count += 1
                    failed_count = 0
                else:
                    success_count = 0
                    failed_count += 1

                if failed_count >= 100:
                    self.print("⚠️ Model stagnated. Perturbing weights.")
                    torch.save(self.model.state_dict(), "backup_model.pth")
                    self.perturb_model_weights(strength=0.02)
                    failed_count = 0

                x_norm = float(data['x_norm'])
                y_norm = float(data['y_norm'])
                loss, output = self.learn_from_feedback(
                    image, task, x_norm, y_norm, correct_done, correct_confidence)

                total_loss += loss.item()
                live.update(self.generate_live_table2(
                    accuracy, acc_done, acc_tap, acc_skip, count, total_loss / count, output))

        self.model.save()

    def learn_from_feedback(self, image_path, task: Task, x, y, done_flag, confidence_flag):
        self.model.train()
        image = Image.open(image_path).convert("L")
        image_tensor, _, _ = ResizeImage.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        task_tensor = torch.tensor(
            [task.value], dtype=torch.long).to(self.device)
        x_y, done_score, confidence = self.model(image_tensor, task_tensor)
        output = torch.cat([x_y, done_score.unsqueeze(1),
                           confidence.unsqueeze(1)], dim=1)

        target = torch.tensor(
            [[x, y, done_flag, confidence_flag]], dtype=torch.float).to(self.device)

        loss_x = F.mse_loss(x_y[0][0], target[0][0])
        loss_y = F.mse_loss(x_y[0][1], target[0][1])
        loss_done = F.binary_cross_entropy_with_logits(
            done_score[0], target[0][2])
        loss_conf = F.mse_loss(confidence[0], target[0][3])

        loss = 0.5 * (loss_x + loss_y) + 0.2 * loss_done + 0.1 * loss_conf
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()

        self.feedback_counter += 1
        if self.feedback_counter % 50 == 0:
            self.model.save()

        loss_str = f"{loss_x.item():.4f}, {loss_y.item():.4f}, {loss_done.item():.4f}, {loss_conf.item():.4f}"
        return loss, loss_str

    def generate_live_table2(self, accuracy, acc_done, acc_tap, acc_skip, iteration, loss, output):
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_row("Device:", f"[bold cyan]{self.device}[/bold cyan]")
        table.add_row("Iteration", f"{iteration}")
        table.add_row("Avg_Loss", f"{loss:.2f}")
        table.add_row("Loss: ", f"{output}")
        table.add_row("Avg Accuracy: ", f"{accuracy:.2f}")
        table.add_row("✔ Tap Accuracy", f"{acc_tap:.2f}")
        table.add_row("✔ Skip Accuracy", f"{acc_skip:.2f}")
        table.add_row("✔ Done Accuracy", f"{acc_done:.2f}")
        return Panel(table, title="Current Action Info", border_style="blue")

    def avg(self, buf):
        return sum(buf) / len(buf) if buf else 0

    def perturb_model_weights(self, strength=0.05):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                if random.random() < 0.2:
                    param.data += torch.randn_like(param) * strength

    def print(self, msg):
        print(f"[Model] {msg}")


# === Load image paths ===
augmented_data_path = os.path.join("augmented_data", "data")
augmented_data = os.path.join(augmented_data_path, f"augmented_data.csv")

train_data_path = os.path.join("train_data", "data")
train_data = os.path.join(train_data_path, "clean_data.csv")

data_sets = [train_data, augmented_data]

all_rows, tap_row, skip_row, done_row = [], [], [], []

for path in data_sets:
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # Clean headers
        if reader.fieldnames is not None:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
        else:
            raise ValueError("CSV has no headers; fieldnames is None")

        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                row['x_norm'] = float(row['x_norm'])
                row['y_norm'] = float(row['y_norm'])
                row['done'] = float(row['done'])
                row['confidence'] = float(row['confidence'])
                all_rows.append(row)
                if row['confidence'] >= 0.9:
                    tap_row.append(row)
                if row['confidence'] <= 0.1:
                    skip_row.append(row)
                if row['done'] >= 0.9:
                    done_row.append(row)
            except Exception as e:
                print(f"Skipping invalid row: {row} -- {e}")

# === Run Training ===
process = Learn()
process.learn()
