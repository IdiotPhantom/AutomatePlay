from setting import *
from adb import ADB
import subprocess
import time


class BlueStack:
    def __init__(self, instance_name):
        self.instance_name = instance_name
        self.start_instance()
        self.connect_adb()

    def __del__(self):
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

    def click_on_screen(self, image, count=IMAGE_FAILED_TIMEOUT):

        success = False

        while not success:
            success = self.instance_adb.tap_button_on_screen(image)

            if success:
                return True

            self.print(f"==>Countdown:{count}  ", end='\r')

            if count > 0:
                count -= 1
                time.sleep(1)
            else:
                raise Exception(
                    f"Failed to find {os.path.basename(image)} on screen")

    def click_on_screen_with_multiple_choice(self, goal_images, images, count=IMAGE_FAILED_TIMEOUT):

        success = False

        while not success:
            if self.find_and_click_images(goal_images):
                return True

            self.find_and_click_images(images)

            self.print(f"==>Countdown:{count}  ", end='\r')
            if count > 0:
                count -= 1
                time.sleep(1)
            else:
                raise Exception(
                    f"Failed to find relevant images on screen")

    def find_and_click_images(self, images):
        if len(images) == 0:
            raise Exception("Nothing image was parse to find")

        for image in images:
            success = self.instance_adb.tap_button_on_screen(image)
            if success:
                return True

        return False

    def game_login(self, throw=True):
        self.print(f"Attempting to click Login btn")
        try:
            self.click_on_screen_with_multiple_choice(
                [START_GAME_IMAGE], [GAME_ICON_IMAGE, LOGIN_IMAGE], GAME_START_TIMEOUT)
        except Exception as e:
            self.print(f"Failed to Login: {e}")
            if throw:
                raise e

    def game_switch_account(self, login_details, throw=True):

        try:
            if login_details["account"] == "google":
                self.print(f"Attempting to click login with google")
                self.click_on_screen_with_multiple_choice([GOOGLE_LOGIN_IMAGE], [
                                                          SWITCH_ACCOUNT_IMAGE, SWITCH_ACCOUNT_LOGIN_IMAGE], count=GAME_START_TIMEOUT)

                self.print(f"Attempting to click google account")
                self.click_on_screen(login_details["password"])

            else:
                self.print(f"Attempting to click login with member")
                self.click_on_screen_with_multiple_choice([MEMBER_LOGIN_IMAGE], [
                                                          SWITCH_ACCOUNT_IMAGE, SWITCH_ACCOUNT_LOGIN_IMAGE], count=GAME_START_TIMEOUT)

                time.sleep(1)

                self.print(f"Attempting to key in details")
                self.instance_adb.adb_tap(*ACCOUNT_TEXT_FIELD_POS)
                self.instance_adb.adb_type(login_details["account"])
                self.instance_adb.adb_tap(*PASSWORD_TEXT_FIELD_POS)
                self.instance_adb.adb_type(login_details["password"])

                self.print(f"Attempting to click login btn")
                self.click_on_screen(MEMBER_LOGIN_ACCOUNT_IMAGE)
        except Exception as e:
            self.print(f"Failed to switch account: {e}")
            if throw:
                raise e

    def game_logout(self, throw=True):
        try:
            self.navigate_to_main_page(throw)

            time.sleep(1)

            self.print(f"Attempting to open setting page")
            self.instance_adb.adb_tap(*SETTING_TAB_POS)

            self.print(f"Attempting to click logout btn")
            self.click_on_screen(LOGOUT_IMAGE, MAX_WULIN_TIMEOUT)
        except Exception as e:
            self.print(f"Failed to logout: {e}")
            if throw:
                raise e

    def navigate_to_main_page(self, throw=True):
        try:
            self.print(f"Attempting to navigate to main page")
            self.click_on_screen_with_multiple_choice([MAIN_PAGE_CHECKER_IMAGE], [
                EXIT_IMAGE, CHAT_CLOSER_IMAGE, MAIL_CLOSER_IMAGE, ACTIVITY_EXIT_IMAGE], BACK_TO_MAIN_PAGE_FAILED_COUNT)
        except Exception as e:
            self.print(f"Failed to Navigate to main page: {e}")
            if throw:
                raise e

    def start_wulin(self, throw=True):

        self.print(f"Attempting to click Acticity btn")
        try:
            self.click_on_screen_with_multiple_choice([ACTIVITY_IMAGE, ACTIVITY2_IMAGE],
                                                      [EXIT_IMAGE], IMAGE_FAILED_TIMEOUT)

            self.print(f"Attempting to click Acticity Icon btn")
            self.click_on_screen(ACTIVITYICON_IMAGE, IMAGE_FAILED_TIMEOUT)

            self.print(f"Attempting to click Join Acticity btn")
            self.click_on_screen(JOINACTIVITY_IMAGE, IMAGE_FAILED_TIMEOUT)

            self.print(f"Attempting to click Start Acticity btn")
            self.click_on_screen(STARTACTIVITY_IMAGE, IMAGE_FAILED_TIMEOUT)
        except Exception as e:
            self.print(f"Failed to Start Activity: {e}")
            if throw:
                raise e

    def collect_rewards(self, throw=True):

        try:
            self.print(f"Attempting to collect Exp")
            try:
                if self.click_on_screen_with_multiple_choice(
                        [COLLECT_EXP_IMAGE], [OFFLINE_EXP_TAB_IMAGE], IMAGE_FAILED_TIMEOUT):
                    self.print(f"Offline Exp Collected")
            except Exception as e:
                self.print(f"Failed to collect Exp:{e}")

            self.print(f"Attempting to collect money")
            try:
                if self.click_on_screen_with_multiple_choice(
                        [COLLECT_MONEY_IMAGE], [MONEY_COLLECTION_TAB_IMAGE], IMAGE_FAILED_TIMEOUT):
                    self.print(f"Money Collected")
            except Exception as e:
                self.print(f"Failed to collect money:{e}")

            self.instance_adb.adb_swipe(170, 400, 170, 200)

            self.print(f"Attempting to retrive missed rewards")
            try:
                if self.click_on_screen_with_multiple_choice(
                        [RETRIVE_REWARD_TAB_IMAGE], [MISSED_REWARD_TAB_IMAGE], IMAGE_FAILED_TIMEOUT):
                    self.print(f"Navigated to retrive reward tab")

                count = 0
                while (MISSED_REWARD_MAX_COUNT > count):
                    count += 1
                    self.print(f"==>Try Count: {count}", end='\r')
                    if self.click_on_screen_with_multiple_choice(
                            [CONFIRM_IMAGE], [RETRIVE_IMAGE], IMAGE_FAILED_TIMEOUT):
                        continue
                    break
                self.print(f"Rewards Collected")
            except Exception as e:
                self.print(f"Failed to collect reward:{e}")

            try:
                self.navigate_to_main_page(throw)
            except Exception as e:
                self.print(f"Failed to navigate to main page:{e}")

            self.countdown(2)   # sleep to let UI to close

            self.print(f"Attempting to retrive rewards in mail")
            self.instance_adb.adb_tap(*OPEN_CHAT_POS)
            try:
                while self.click_on_screen_with_multiple_choice(
                        [COLLECT_ATTACHMENT_IMAGE], [MAIL_CONTANIER_IMAGE, MAILBOX_IMAGE], IMAGE_FAILED_TIMEOUT):
                    self.print(f"Mail Collected")

            except Exception as e:
                self.print(f"Failed to collect mails: {e}")
                self.print(f"Attempting to delete mails")
                try:
                    if self.click_on_screen_with_multiple_choice(
                            [CONFIRM_DELETE_MAILS_IMAGE], [DELETE_MAILS_IMAGE], IMAGE_FAILED_TIMEOUT):
                        self.print(f"Mails deleted")
                except Exception as e2:
                    self.print(f"Failed to delete mails: {e2}")

        except Exception as e:
            self.print(f"Failed to collect:{e}")
            if throw:
                raise e

    def countdown(self, count):
        while count > 0:
            self.print(f"==>Countdown:{count}  ", end='\r')
            count -= 1
            time.sleep(1)

        self.print(' ' * 30, end='\r')

    def print(self, message, **kwargs):
        print(f"\t[+]{message}", flush=True, **kwargs)

    def automate_wulin(self):

        self.game_login()
        self.start_wulin()

        self.countdown(INTERVAL)    # window for running activity

    def automatic_collect_rewards(self):

        self.game_login()
        self.collect_rewards()

    def automate_multiple_wulin(self, login_details):

        for login_detail in login_details:
            self.game_switch_account(login_detail, throw=False)
            self.game_login(throw=False)
            self.start_wulin(throw=False)
            self.game_logout(throw=False)

    def automatic_multiple_collect_rewards(self, login_details):

        for login_detail in login_details:
            self.game_switch_account(login_detail, throw=False)
            self.game_login(throw=False)
            self.collect_rewards(throw=False)
            self.game_logout(throw=False)
