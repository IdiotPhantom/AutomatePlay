import os
import time
import csv
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

from GameActionModel import GameActionModel
from resize_image import ResizeImage
from setting import Task, SCREEN_HEIGHT, SCREEN_WIDTH,  TRAINING_DATA_NAME

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.console import Console


# ==== Constants ====
LOW_CONFIDENCE_THRESHOLD = 0.2
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
DONE_SCORE_THRESHOLD = 10
CIRCLE_RADIUS = 15


class Learn:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = GameActionModel(num_tasks=len(Task)).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.feedback_counter = 0
        self.loss_history = []

    def start_learn(self, end_count=10, data=[]):
        trigger = 0
        success_count = 0
        count = 0
        accumulated_confidence = 0

        with Live(console=Console(), refresh_per_second=4) as live:
            while trigger < 0.7:
                time.sleep(0.1)
                count += 1
                accumulated_confidence += trigger

                rand = random.randint(0, len(data) - 1)
                train_data = data[rand]

                image_path, task_id, x_data, y_data, done_data, confidence_data = train_data

                if not os.path.exists(image_path):
                    self.print(f"[WARN] Image not found: {image_path}")
                    continue

                action, loaded_image = self.decide_action(image_path, task_id)
                if action["type"] == "done":
                    self.print(f"[{Task(task_id).name}] Task is completed.")
                    break

                elif action["type"] == "tap":
                    confidence = action["confidence"]
                    trigger = confidence

                    if trigger < LOW_CONFIDENCE_THRESHOLD:
                        x, y = random_point_in_circle(x_data, y_data)
                        x_norm = x / SCREEN_WIDTH
                        y_norm = y / SCREEN_HEIGHT
                        self.print(
                            "🔄 Super low confidence — using random tap within range.")
                    elif trigger < MEDIUM_CONFIDENCE_THRESHOLD:
                        x = random.randint(0, SCREEN_WIDTH - 1)
                        y = random.randint(0, SCREEN_HEIGHT - 1)
                        x_norm = x / SCREEN_WIDTH
                        y_norm = y / SCREEN_HEIGHT
                        self.print("🔄 Low confidence — using random tap.")
                    else:
                        x, y = action["x"], action["y"]
                        x_norm, y_norm = action["x_norm"], action["y_norm"]
                        self.print(f"✨ Confidence — tap on ({x}, {y})")

                    # Check if tap is correct
                    correct = is_point_in_circle(
                        x, y, x_data, y_data, CIRCLE_RADIUS)
                    if correct:
                        success_count += 1
                        confidence_flag = 1.0
                        self.print("✅ Tap is within range.")
                    else:
                        success_count = 0
                        confidence_flag = 0.2
                        self.print("❌ Tap is out of range.")

                    feedback = self.record_feedback(
                        image_path, task_id, x_norm, y_norm, confidence_flag=confidence_flag
                    )

                    loss = self.learn_from_feedback(
                        *feedback, loaded_image=loaded_image)

                    live.update(self.generate_live_table(
                        Task(
                            task_id).name, confidence, x, y, image_path, count, accumulated_confidence, loss
                    ))

                    if success_count > end_count:
                        break
                else:
                    self.print(
                        f"[{Task(task_id).name}] Skipped: {action.get('reason', 'unknown')}")

    def decide_action(self, image_path, task: int):
        image = Image.open(image_path).convert("RGB")
        tensor_image, scale, padding = ResizeImage.preprocess_image(image)
        tensor_image = tensor_image.to(self.device)

        task_tensor = torch.tensor([task], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_image, task_tensor)

        x_norm, y_norm, done_score, confidence = output[0]
        x, y = ResizeImage.unpad_and_rescale_coords(
            x_norm.item(), y_norm.item(), scale, padding,
            orig_w=SCREEN_WIDTH, orig_h=SCREEN_HEIGHT
        )

        if done_score.item() > DONE_SCORE_THRESHOLD:
            return {"type": "done"}, image

        return {
            "type": "tap",
            "x": x,
            "y": y,
            "x_norm": x_norm.item(),
            "y_norm": y_norm.item(),
            "confidence": confidence.item()
        }, image

    def record_feedback(self, image_path, task: int, x_norm, y_norm, confidence_flag=1.0):
        done = 0
        return (image_path, task, x_norm, y_norm, done, confidence_flag)

    def learn_from_feedback(self, image_path, task: int, x, y, done, confidence_flag, loaded_image=None):
        self.model.train()

        x = float(x)
        y = float(y)
        if x > 1 or y > 1:
            x /= SCREEN_WIDTH
            y /= SCREEN_HEIGHT

        image = loaded_image or Image.open(image_path).convert("RGB")
        image_tensor, _, _ = ResizeImage.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        task_tensor = torch.tensor([task], dtype=torch.long).to(self.device)
        label = torch.tensor([[x, y, float(done), float(
            confidence_flag)]], dtype=torch.float).to(self.device)

        output = self.model(image_tensor, task_tensor)
        loss = self.criterion(output, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.feedback_counter += 1

        if self.feedback_counter % 10 == 0:
            self.print("💾 Saving model...")
            self.model.save()

        self.loss_history.append(loss.item())  # store loss
        return loss.item()  # return latest loss

    def generate_live_table(self, task_name, confidence, tap_x, tap_y, image_path, iteration, accumulated_data, loss):
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        avg_conf = (accumulated_data / iteration) if iteration > 0 else 0

        table.add_row("Task:", f"[bold cyan]{task_name}[/bold cyan]")
        table.add_row(
            "Confidence:", f"[bold green]{confidence:.2f}[/bold green]")
        table.add_row("Tap Position:",
                      f"[bold magenta]({tap_x}, {tap_y})[/bold magenta]")
        table.add_row("Image:", f"{os.path.basename(image_path)}")
        table.add_row("Iteration", f"{iteration}")
        table.add_row("Avg Confidence", f"{avg_conf:.2f}")
        table.add_row("Loss:", f"[bold red]{loss:.6f}[/bold red]")

        return Panel(table, title="Current Action Info", border_style="blue")

    def plot_loss_curve(self, filename="loss_curve.png"):
        if not self.loss_history:
            print("[WARN] No loss history to plot.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label="Training Loss", color='red')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        print(f"[INFO] Loss curve saved to {filename}")

    def print(self, message):
        print(f"\t[@] {message}")


def is_point_in_circle(px, py, cx, cy, r):
    dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return dist <= r


def random_point_in_circle(cx, cy, radius=15, x_min=0, x_max=SCREEN_WIDTH, y_min=0, y_max=SCREEN_HEIGHT):
    for _ in range(100):
        dx = random.uniform(-radius, radius)
        dy = random.uniform(-radius, radius)
        if dx * dx + dy * dy <= radius * radius:
            x = cx + dx
            y = cy + dy
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return x, y
    raise ValueError("Unable to find valid point in circle.")


# === Load CSV Data ===
data = []
with open(TRAINING_DATA_NAME, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            image_path = row[0]
            task_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            done = int(row[4])
            confidence = float(row[5])
            data.append((image_path, task_id, x, y, done, confidence))
        except Exception as e:
            print(f"[WARN] Failed to load row: {row} — {e}")

unique_images = set(row[0] for row in data)
print(f"[INFO] Loaded {len(data)} data points.")

process = Learn()
process.start_learn(data=data)
process.plot_loss_curve()
