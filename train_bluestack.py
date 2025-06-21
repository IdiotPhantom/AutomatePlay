from setting import IMAGE_FAILED_TIMEOUT, Task, PLAYER_EXE, GAME_PACKAGE, INSTANCE_STARTING_TIME, ADB_PATH
import os
from train_adb import TrainADB
from train_set import TrainSet
import subprocess
import time


class TrainBlueStack:
    def __init__(self, instance_name, terminate_on_destroy=True, count=INSTANCE_STARTING_TIME):
        self.instance_name = instance_name
        self.start_instance(count)
        self.connect_adb()
        self.terminate_on_destory = terminate_on_destroy

    def __del__(self):
        if self.terminate_on_destory:
            self.terminate_instance()

    def start_instance(self, count=INSTANCE_STARTING_TIME):
        try:
            self.print(f"Starting Instance: {self.instance_name}")
            subprocess.Popen([PLAYER_EXE, "--instance", self.instance_name,
                              "--cmd", "launchApp", "--package", GAME_PACKAGE])

            # waiting for adb to start up
            time.sleep(count)
        except subprocess.CalledProcessError as e:
            raise e

    def terminate_instance(self):
        try:
            self.print(f"Closing Instance: {self.instance_name}")
            subprocess.run(["taskkill", "/F", "/IM", "HD-Player.exe"])
        except subprocess.CalledProcessError as e:
            raise e

    def connect_adb(self):
        self.instance_adb = TrainADB(ADB_PATH)
        if not self.instance_adb.adb_get_port():
            raise Exception("Failed to find port")

        if not self.instance_adb.adb_connect():
            raise Exception("Failed to connect port")

    def click_on_screen(self, image, count=IMAGE_FAILED_TIMEOUT, task: Task = None):

        success = False

        while not success:
            success = self.instance_adb.tap_button_on_screen(image)

            if success is not None:
                x = success["x"]
                y = success["y"]
                TrainSet.capture_screen(task, x, y)
                return True

            TrainSet.capture_not_confident_screen(task)

            if count > 0:
                count -= 1
                time.sleep(1)
            else:
                raise Exception(
                    f"Failed to find {os.path.basename(image)} on screen")

    def click_on_screen_to_check(self, goal_images, images, count=IMAGE_FAILED_TIMEOUT, task: Task = None):

        success = False

        while not success:
            if self.find_images(goal_images, task, record=False, click=False):
                return True

            self.find_images(images, task)

            if count > 0:
                count -= 1
                time.sleep(1)
            else:
                raise Exception(
                    f"Failed to find relevant images on screen")

    def find_images(self, images, task=None, record=True, click=True):
        if len(images) == 0:
            return

        for image in images:
            success = self.instance_adb.tap_button_on_screen(image, tap=click)
            if success is not None:
                x = success["x"]
                y = success["y"]
                if record:
                    TrainSet.capture_screen(task, x, y)
                return True

            TrainSet.capture_not_confident_screen(task)

        return False

    def print(self, message, **kwargs):
        print(f"\t[-]{message}", flush=True, **kwargs)
