from setting import *
from adb import ADB
import subprocess
import time


class BlueStack:
    def __init__(self, instance_name, terminate_on_destroy=True):
        self.instance_name = instance_name
        self.start_instance()
        self.connect_adb()
        self.terminate_on_destory = terminate_on_destroy

    def __del__(self):
        if self.terminate_on_destory:
            self.terminate_instance()

    def start_instance(self):
        try:
            self.print(f"Starting Instance: {self.instance_name}")
            subprocess.Popen([PLAYER_EXE, "--instance", self.instance_name,
                              "--cmd", "launchApp", "--package", GAME_PACKAGE])

            # waiting for adb to start up
            self.countdown(INSTANCE_STARTING_TIME)
        except subprocess.CalledProcessError as e:
            raise e

    def terminate_instance(self):
        try:
            self.print(f"Closing Instance: {self.instance_name}")
            subprocess.run(["taskkill", "/F", "/IM", "HD-Player.exe"])
        except subprocess.CalledProcessError as e:
            raise e

    def connect_adb(self):
        self.instance_adb = ADB(ADB_PATH)
        if not self.instance_adb.adb_get_port():
            raise Exception("Failed to find port")

        if not self.instance_adb.adb_connect():
            raise Exception("Failed to connect port")

    def click_on_screen(self, image, count=IMAGE_FAILED_TIMEOUT, screen_record=True):

        success = False

        while not success:
            success = self.instance_adb.tap_button_on_screen(
                image, screen_record)

            if success:
                return True

            self.print(f"==>Countdown:{count}  ", end='\r')

            if count > 0:
                count -= 1
                time.sleep(1)
            else:
                raise Exception(
                    f"Failed to find {os.path.basename(image)} on screen")

    def click_on_screen_with_multiple_choice(self, goal_images, images, count=IMAGE_FAILED_TIMEOUT, screen_record=True):

        success = False

        while not success:
            if self.find_and_click_images(goal_images, screen_record):
                return True

            self.find_and_click_images(images, screen_record)

            self.print(f"==>Countdown:{count}  ", end='\r')
            if count > 0:
                count -= 1
                time.sleep(1)
            else:
                raise Exception(
                    f"Failed to find relevant images on screen")

    def find_and_click_images(self, images, screen_record=True):
        if len(images) == 0:
            raise Exception("Nothing image was parse to find")

        for image in images:
            success = self.instance_adb.tap_button_on_screen(
                image, screen_record)
            if success:
                return True

        return False

    def countdown(self, count):
        while count > 0:
            self.print(f"==>Countdown:{count}  ", end='\r')
            count -= 1
            time.sleep(1)

        self.print(' ' * 30, end='\r')

    def print(self, message, **kwargs):
        print(f"\t[+]{message}", flush=True, **kwargs)
