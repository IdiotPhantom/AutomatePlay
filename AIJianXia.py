from PIL import Image
import torch
import time
import torchvision.transforms as transforms
from GameActionModel import GameActionModel
from bluestack import BlueStack
from setting import SCREEN_IMAGE, SCREEN_WIDTH, SCREEN_HEIGHT, Task, MASTER_INSTANCE, CONFIDENCE_THRESHOLD


class AIJianXia:
    def __init__(self, instance_name=MASTER_INSTANCE):
        # Load model
        self.model = GameActionModel(num_tasks=len(Task))
        self.model.eval()

        # Start Instance
        self.print(f"Starting JianXia in instance: {instance_name}")

        self.instance = BlueStack(instance_name, False)

    def start_loop(self, task: Task):
        while True:
            time.sleep(1)
            action = self.decide_action(SCREEN_IMAGE, task)

            if action["type"] == "done":
                print(f"[{task.name}] task is completed.")
                break
            elif action["type"] == "tap":
                confidence = action["confidence"]
                if confidence < CONFIDENCE_THRESHOLD:
                    print(
                        f"[{task.name}] low confidence ({confidence:.2f}), skipping tap")
                    continue

                x, y = action["x"], action["y"]
                x_norm, y_norm = action["x_norm"], action["y_norm"]

                print(
                    f"[{task.name}] tapping at ({x}, {y}), ({x_norm:.2f},{y_norm:.2f}) with confidence {confidence:.2f}"
                )
                self.instance.instance_adb.adb_tap(x, y)
            else:
                print(
                    f"[{task.name}] skipped due to reason: {action.get('reason', 'unknown')}")


    def decide_action(self, screenshot_path, task: Task):
        self.instance.instance_adb.take_screenshot()

        # Load and preprocess image
        image = Image.open(screenshot_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        tensor_image = transform(image).unsqueeze(0)  # Add batch dimension

        # Convert task enum to tensor
        task_id = torch.tensor([task.value], dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            output = self.model(tensor_image, task_id)

        # Get values
        x_norm, y_norm, done_score, confidence = output[0]

        x = int(x_norm.item() * SCREEN_WIDTH)
        y = int(y_norm.item() * SCREEN_HEIGHT)

        # Interpret result
        if done_score.item() > 0.5:
            return {"type": "done"}
        else:
            return {"type": "tap", "x": x, "y": y, "x_norm": x_norm, "y_norm": y_norm, "confidence": confidence}

    def print(self, message):
        print(f"\t[@]{message}")


task = Task.JOIN_TEAM
model = AIJianXia()
model.start_loop(task)
