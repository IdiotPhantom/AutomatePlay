from setting import Task, TRAINING_IMAGE_PATH, TRAINING_DATA_NAME, SCREEN_HEIGHT, SCREEN_WIDTH
import time
import cv2
import os


class TrainSet:
    def capture_not_confident_screen(task: Task):
        if task is None:
            return

        __class__.print(f"capturing data: Not Confident")

        screen = cv2.imread("screen.png")
        if screen is None:
            raise ValueError("Failed to load screenshot.")

        directory = TRAINING_IMAGE_PATH
        os.makedirs(directory, exist_ok=True)

        screenshot_path = os.path.join(
            directory, f"screenshot_{int(time.time())}.png")
        cv2.imwrite(screenshot_path, screen)

        # Save with dummy x/y, done = 0, confidence = 0
        with open(TRAINING_DATA_NAME, "a") as f:
            f.write(
                f"{screenshot_path},{task.value},0.0000,0.0000,0,0\n")

    def capture_done_screen(task):
        if task is None:
            return

        __class__.print(f"capturing data: Done")

        screen = cv2.imread("screen.png")
        if screen is None:
            raise ValueError("Failed to load screenshot.")

        directory = TRAINING_IMAGE_PATH
        os.makedirs(directory, exist_ok=True)

        screenshot_path = os.path.join(
            directory, f"screenshot_{int(time.time())}.png")
        cv2.imwrite(screenshot_path, screen)

        # Save with dummy x/y, done = 1, confidence = 1
        with open(TRAINING_DATA_NAME, "a") as f:
            f.write(
                f"{screenshot_path},{task.value},0.0000,0.0000,1,1\n")

    def capture_screen(task, x, y):
        if task is None:
            return

        __class__.print(f"capturing data: Confident")

        screen = cv2.imread("screen.png")
        if screen is None:
            raise ValueError("Failed to load screenshot.")

        directory = TRAINING_IMAGE_PATH
        os.makedirs(directory, exist_ok=True)

        screenshot_path = os.path.join(
            directory, f"screenshot_{int(time.time())}.png")
        cv2.imwrite(screenshot_path, screen)

        # Normalize x/y
        x_norm = x / SCREEN_WIDTH
        y_norm = y / SCREEN_HEIGHT

        # Save tap label: done = 0, confidence = 1
        with open(TRAINING_DATA_NAME, "a") as f:
            f.write(
                f"{screenshot_path},{task.value},{x_norm:.4f},{y_norm:.4f},0,1\n")

    def print(message):
        print(f"\t\t[@]{message}")
