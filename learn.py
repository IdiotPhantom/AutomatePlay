import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from GameActionModel import GameActionModel
from resize_image import ResizeImage
from setting import Task, SCREEN_HEIGHT, SCREEN_WIDTH, LEARNING_IMAGE_PATH, SCREEN_IMAGE, SCRIPT_DIR, TRAINING_DATA_PATH

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console


class Learn:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = GameActionModel(num_tasks=len(Task)).to(self.device)
        self.model.eval()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.feedback_counter = 0

        for param in self.model.cnn.parameters():
            param.requires_grad = True

    def decide_action(self, image_path, task: Task):
        image = Image.open(image_path).convert("RGB")
        tensor_image, scale, padding = ResizeImage.preprocess_image(image)

        task_tensor = torch.tensor(
            [task.value], dtype=torch.long).to(self.device)
        tensor_image = tensor_image.to(self.device)

        with torch.no_grad():
            output = self.model(tensor_image, task_tensor)

        x_norm, y_norm, done_score, confidence = output[0]
        x, y = ResizeImage.unpad_and_rescale_coords(
            x_norm.item(), y_norm.item(),
            scale, padding,
            orig_w=SCREEN_WIDTH, orig_h=SCREEN_HEIGHT
        )

        if done_score.item() > 10:  # Threshold for task completion
            return {"type": "done"}

        return {
            "type": "tap",
            "x": x,
            "y": y,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "confidence": confidence
        }

    def learn(self, datas):
        success_count = 0
        failed_count = 0
        correct_count = 0
        while (correct_count/100) < 0.8:
            rand = random.randrange(0, len(datas)-1)
            data = datas[rand]
            image = data['image_path']
            task = Task(int(data['task']))
            correct_x_norm = data['x_norm']
            correct_y_norm = data['y_norm']
            correct_x = correct_x_norm * SCREEN_WIDTH
            correct_y = correct_y_norm * SCREEN_HEIGHT
            correct_done = data['done']
            correct_confidence = data['confidence']
            action = self.decide_action(image, task)

            if action["type"] == "done":
                if correct_done == 1:
                    correct_count += 1
                    success_count += 1
                    failed_count = 0  # ✅ Reset failures if success
                    self.print("✅ Correct done.")
                else:
                    success_count = 0
                    failed_count += 1
                    self.print("❌ Incorrect done.")

            elif action["type"] == "tap":

                x, y = action["x"], action["y"]

                correct = is_within_radius(x, y, correct_x, correct_y)

                if correct:
                    correct_count += 1
                    x_feedback = x  # model was right
                    y_feedback = y
                    success_count += 1
                    failed_count = 0  # ✅ Reset failures if success
                    self.print("✅ Tap is within target zone.")
                else:
                    success_count = 0
                    x_feedback = (x + correct_x) / 2
                    y_feedback = (y + correct_y) / 2
                    success_count = max(success_count - 1, 0)
                    self.print(
                        "❌ Tap is out of range. Using center correction.")

            else:
                self.print(
                    f"[{task.name}] Skipped: {action.get('reason', 'unknown')}")
                continue

            failed_count += 1
            if failed_count >= 10:  # ✅ Perturb model after 10 failures
                self.perturb_model_weights(strength=0.005)
                failed_count = 0

            x_norm = x_feedback / SCREEN_WIDTH
            y_norm = y_feedback / SCREEN_HEIGHT
            done_flag = correct_done
            confidence_flag = correct_confidence
            self.learn_from_feedback(
                image, task, x_norm, y_norm, done_flag, confidence_flag)

            self.print(f"Correctness: {(correct / 100):.2f}")

    def learn_from_feedback(self, image_path, task: Task, x, y, done_flag, confidence_flag):
        self.model.train()

        # Normalize target if needed
        if x > 1 or y > 1:
            x /= SCREEN_WIDTH
            y /= SCREEN_HEIGHT

        # Prepare input
        image = Image.open(image_path).convert("RGB")
        image_tensor, _, _ = ResizeImage.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        task_tensor = torch.tensor(
            [task.value], dtype=torch.long).to(self.device)

        # Prepare target
        target = torch.tensor([[x, y, float(done_flag), float(
            confidence_flag)]], dtype=torch.float).to(self.device)

        # Forward pass
        output = self.model(image_tensor, task_tensor)

        # Loss components
        loss_x = F.mse_loss(output[0][0], target[0][0])
        loss_y = F.mse_loss(output[0][1], target[0][1])
        loss_done = F.binary_cross_entropy_with_logits(
            output[0][2], target[0][2])
        loss_conf = F.mse_loss(output[0][3], target[0][3])

        # Total loss
        loss = 1.0 * (loss_x + loss_y) + 0.1 * loss_done + 0.2 * loss_conf

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.model.eval()
        self.feedback_counter += 1

        if self.feedback_counter % 50 == 0:
            self.print("💾 Saving model...")
            self.model.save()

        return loss

    def perturb_model_weights(self, strength=0.05):
        self.print("🔧 Perturbing model weights to escape local minima...")
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                if random.random() < 0.2:  # 20% of layers/neurons
                    noise = torch.randn_like(param) * strength
                    param.data += noise

    def print(self, message):
        print(f"\t[@] {message}")


def is_within_radius(x_pred, y_pred, x_target, y_target, radius=15):
    return (abs(x_pred - x_target) <= radius) and (abs(y_pred - y_target) <= radius)


# === Load image paths ===
augmented_data_path = os.path.join(SCRIPT_DIR, "augmented_data")
augmented_data = os.path.join(augmented_data_path, f"augmented_data.csv")
training_data = os.path.join(TRAINING_DATA_PATH, "data2025-06-21.csv")
data_sets = [augmented_data, training_data]

all_rows = []

for data_path in data_sets:
    with open(data_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=[
            'image_path', 'task', 'x_norm', 'y_norm', 'done', 'confidence'])
        for row in reader:
            # Convert or process if needed
            row['x_norm'] = float(row['x_norm'])
            row['y_norm'] = float(row['y_norm'])
            row['done'] = float(row['done'])
            row['confidence'] = float(row['confidence'])
            # Just keep row as dict, or do other processing

            # Append to your list, NOT the file path string
            all_rows.append(row)


process = Learn()
process.learn(all_rows)
