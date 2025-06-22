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

        for param in self.model.cnn.parameters():
            param.requires_grad = True

    def decide_action(self, image_path, task: Task, return_raw_output=False):

        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert("L")

        except Exception as e:
            print(f"[!] Error loading image: {image_path} -- {e}")
            raise  # Optional: re-raise to stop execution or allow higher-level handler to catch it

        tensor_image, scale, padding = ResizeImage.preprocess_image(image)

        task_tensor = torch.tensor(
            [task.value], dtype=torch.long).to(self.device)
        tensor_image = tensor_image.to(self.device)

        self.model.eval()

        with torch.no_grad():
            output = self.model(tensor_image, task_tensor)

        x_norm, y_norm, done_score, confidence = output[0]
        x, y = ResizeImage.unpad_and_rescale_coords(
            x_norm.item(), y_norm.item(),
            scale, padding,
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
            action["raw_output"] = output

        return action

    def learn(self):

        success_count = 0
        failed_count = 0

        total_loss = 0
        count = 0

        # Seed all with 1/3 chance
        self.acc_done.extend([0, 1])
        self.acc_skip.extend([0, 1])
        self.acc_tap.extend([0, 1])

        with Live(console=Console(), refresh_per_second=4) as live:
            while True:
                if keyboard.is_pressed('esc'):
                    print("ESC pressed, exiting loop.")
                    break

                count += 1

                # Compute recent accuracy using the buffer
                accuracy = self.avg(self.correct_buffer)
                acc_done = self.avg(self.acc_done)
                acc_skip = self.avg(self.acc_skip)
                acc_tap = self.avg(self.acc_tap)

                if accuracy < 0.1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 3e-3
                elif accuracy < 0.3:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 1e-3
                elif accuracy < 0.6:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 1e-4
                elif accuracy > 0.6:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 1e-5
                elif accuracy > 0.9:
                    break

                choice = min(acc_tap, acc_done, acc_skip, 0.5)
                if choice == acc_tap:
                    rows = tap_row
                elif choice == acc_done:
                    rows == done_row
                elif choice == acc_skip:
                    rows = skip_row

                rand = random.randrange(0, 10)
                if rand > 6:
                    rows = all_rows

                rand = random.randrange(0, len(rows))
                data = rows[rand]
                image = data['image_path']
                task = Task(int(data['task']))
                correct_x_norm = data['x_norm']
                correct_y_norm = data['y_norm']
                correct_x = correct_x_norm * SCREEN_WIDTH
                correct_y = correct_y_norm * SCREEN_HEIGHT
                correct_done = data['done']
                correct_confidence = data['confidence']

                x_feedback = 0
                y_feedback = 0

                action = self.decide_action(image, task)

                is_current_correct = False

                action_type = action["type"]

                if action_type == "done":
                    if correct_done == 1:
                        is_current_correct = True
                        self.print("✅ Correct done.")
                    else:
                        self.print("❌ Incorrect done.")

                    self.acc_done.append(is_current_correct)

                elif action_type == "tap":
                    x, y = action["x"], action["y"]

                    if correct_confidence == 1:
                        correct = is_within_radius(x, y, correct_x, correct_y)
                        if correct:
                            is_current_correct = True
                            x_feedback = x
                            y_feedback = y
                            self.print("✅ Tap is within target zone.")
                        else:
                            x_feedback = correct_x
                            y_feedback = correct_y
                            self.print(
                                "❌ Tap is out of range. Using center correction.")
                    else:
                        self.print("❌ Need to skip instead of tap.")

                    self.acc_tap.append(is_current_correct)

                elif action_type == "skip":
                    if correct_confidence == 0:
                        is_current_correct = True
                        self.print("✅ Correct skip.")
                    else:
                        self.print("❌ Incorrect skip.")

                    self.acc_skip.append(is_current_correct)

                # Append accuracy to buffer
                self.correct_buffer.append(is_current_correct)

                # Update your success_count and failed_count as before
                if is_current_correct:
                    success_count += 1
                    failed_count = 0
                else:
                    success_count = 0
                    failed_count += 1

                if failed_count >= 100:  # ✅ Perturb model
                    self.perturb_model_weights(strength=0.02)
                    failed_count = 0

                x_norm = x_feedback / SCREEN_WIDTH
                y_norm = y_feedback / SCREEN_HEIGHT
                done_flag = correct_done
                confidence_flag = correct_confidence
                loss, output = self.learn_from_feedback(
                    image, task, x_norm, y_norm, done_flag, confidence_flag)

                total_loss += loss.item()

                live.update(
                    self.generate_live_table2(
                        accuracy=accuracy,
                        acc_tap=acc_tap,
                        acc_skip=acc_skip,
                        acc_done=acc_done,
                        iteration=count,
                        loss=(total_loss / count),
                        output=output
                    )
                )

        self.print("Saving Model")
        self.model.save()

    def learn_from_feedback(self, image_path, task: Task, x, y, done_flag, confidence_flag, cached_output=None):

        self.model.train()

        image = Image.open(image_path).convert("L")
        image_tensor, _, _ = ResizeImage.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        task_tensor = torch.tensor(
            [task.value], dtype=torch.long).to(self.device)
        output = self.model(image_tensor, task_tensor)

        # Normalize target if needed
        if x > 1 or y > 1:
            x /= SCREEN_WIDTH
            y /= SCREEN_HEIGHT

        # Prepare target
        target = torch.tensor([[x, y, float(done_flag), float(
            confidence_flag)]], dtype=torch.float).to(self.device)

        # Loss components
        loss_x = F.mse_loss(output[0][0], target[0][0])
        loss_y = F.mse_loss(output[0][1], target[0][1])
        clamped_done_logit = torch.clamp(output[0][2], min=-10, max=10)
        loss_done = F.binary_cross_entropy_with_logits(
            clamped_done_logit, target[0][2])
        loss_conf = F.mse_loss(output[0][3], target[0][3])

        # Total loss
        output = f"{loss_x.item():.4f}, {loss_y.item():.4f}, {loss_done.item():.4f}, {loss_conf.item():.4f}"
        loss = (0.5 * (loss_x + loss_y)
                + 0.2 * loss_done
                + 0.1 * loss_conf)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.model.eval()
        self.feedback_counter += 1

        if self.feedback_counter % 50 == 0:
            self.print("💾 Saving model...")
            self.model.save()

        return loss, output

    def weighted_sample_by_accuracy(self, groups, acc_done, acc_skip, acc_tap):
        # Avoid division by zero and clamp accuracy in a valid range
        epsilon = 1e-6
        acc_done = max(acc_done, epsilon)
        acc_skip = max(acc_skip, epsilon)
        acc_tap = max(acc_tap, epsilon)

        inv_done = 1.0 / acc_done
        inv_skip = 1.0 / acc_skip
        inv_tap = 1.0 / acc_tap

        total = inv_done + inv_skip + inv_tap
        weights = [inv_done / total, inv_skip / total, inv_tap / total]

        group_choice = random.choices(groups, weights=weights)[0]
        return random.choice(group_choice)

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
        return sum(buf) / len(buf) if len(buf) > 0 else 0

    def perturb_model_weights(self, strength=0.05):
        self.print("🔧 Perturbing model weights to escape local minima...")
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                if random.random() < 0.2:  # 20% of layers/neurons
                    noise = torch.randn_like(param) * strength
                    param.data += noise

    def print(self, message):
        print(f"\t[@] {message}")


def is_within_radius(x_pred, y_pred, x_target, y_target, radius=30):
    return (abs(x_pred - x_target) <= radius) and (abs(y_pred - y_target) <= radius)


# === Load image paths ===
augmented_data_path = os.path.join("augmented_data", "data")
augmented_data = os.path.join(augmented_data_path, f"augmented_data.csv")

train_data_path = os.path.join("train_data", "data")
train_data = os.path.join(train_data_path, "clean_data.csv")

data_sets = [train_data, augmented_data]


all_rows = []

for data_set in data_sets:
    with open(data_set, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Clean fieldnames (header)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]

        for row in reader:
            # Ensure it's a dict and clean each key-value
            row = {k.strip(): v.strip() for k, v in row.items()}

            try:
                row['x_norm'] = float(row['x_norm'])
                row['y_norm'] = float(row['y_norm'])
                row['done'] = int(row['done'])
                row['confidence'] = int(row['confidence'])
                all_rows.append(row)
            except Exception as e:
                print(f"[!] Skipping invalid row: {row} -- {e}")
                raise e

done_row = []
tap_row = []
skip_row = []

for row in all_rows:
    if row['done'] == 1:
        done_row.append(row)
    if row['confidence'] == 1:
        tap_row.append(row)
    if row['confidence'] == 0:
        skip_row.append(row)


# Sample one of each category in round-robin
process = Learn()
process.learn()
