from setting import *
from bluestack import BlueStack
import time


class JianXia:
    def __init__(self, instance_name, task: Task = None, label_file=None, terminate_on_destroy=True):
        Train.TRAINING_TASK = None
        self.print(f"Starting JianXia in instance: {instance_name}")

        self.instance = BlueStack(instance_name, terminate_on_destroy)

    def game_login(self, throw=True):
        Train.TRAINING_TASK = Task.LOGIN
        self.print(f"Attempting to click Login btn")
        try:
            self.instance.click_on_screen(LOGIN_IMAGE)
            self.instance.click_on_screen(START_GAME_IMAGE)

            time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
            self.instance.instance_adb.capture_done_screen()
        except Exception as e:
            self.print(f"Failed to Login: {e}")
            if throw:
                raise e

    def game_switch_account(self, login_details, throw=True):
        Train.TRAINING_TASK = None
        try:
            if login_details["account"] == "google":
                self.print(f"Attempting to click login with google")
                self.instance.click_on_screen_with_multiple_choice([GOOGLE_LOGIN_IMAGE], [
                                                                   SWITCH_ACCOUNT_IMAGE, SWITCH_ACCOUNT_LOGIN_IMAGE], count=GAME_START_TIMEOUT)

                self.print(f"Attempting to click google account")
                self.instance.click_on_screen(login_details["password"])

            else:
                self.print(f"Attempting to click login with member")
                self.instance.click_on_screen_with_multiple_choice([MEMBER_LOGIN_IMAGE], [
                                                                   SWITCH_ACCOUNT_IMAGE, SWITCH_ACCOUNT_LOGIN_IMAGE], count=GAME_START_TIMEOUT)

                time.sleep(1)

                self.print(f"Attempting to key in details")
                self.instance.instance_adb.adb_tap(*ACCOUNT_TEXT_FIELD_POS)
                self.instance.instance_adb.adb_type(login_details["account"])
                self.instance.instance_adb.adb_tap(*PASSWORD_TEXT_FIELD_POS)
                self.instance.instance_adb.adb_type(login_details["password"])

                self.print(f"Attempting to click login btn")
                self.instance.click_on_screen(MEMBER_LOGIN_ACCOUNT_IMAGE)
        except Exception as e:
            self.print(f"Failed to switch account: {e}")
            if throw:
                raise e

    def game_logout(self, throw=True):
        try:
            self.navigate_to_main_page(throw)

            Train.TRAINING_TASK = Task.LOGOUT
            self.print(f"Attempting to open setting page")
            self.instance.instance_adb.adb_tap(*SETTING_TAB_POS)
            self.instance.instance_adb.capture_screen(*SETTING_TAB_POS)

            self.print(f"Attempting to click logout btn")
            self.instance.click_on_screen(LOGOUT_IMAGE, MAX_WULIN_TIMEOUT)

            time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
            self.instance.instance_adb.capture_done_screen()
        except Exception as e:
            self.print(f"Failed to logout: {e}")
            if throw:
                raise e

    def navigate_to_main_page(self, throw=True):
        Train.TRAINING_TASK = Task.BACK_TO_MAIN_PAGE
        try:
            self.print(f"Attempting to navigate to main page")
            self.instance.click_on_screen_with_multiple_choice([MAIN_PAGE_CHECKER_IMAGE], [
                EXIT_IMAGE, EXIT_STORE_IMAGE, CHAT_CLOSER_IMAGE, MAIL_CLOSER_IMAGE, ACTIVITY_EXIT_IMAGE, WULIN_DONE_IMAGE], BACK_TO_MAIN_PAGE_FAILED_COUNT, False)

            time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
            self.instance.instance_adb.capture_done_screen()
        except Exception as e:
            self.print(f"Failed to Navigate to main page: {e}")
            if throw:
                raise e

    def start_wulin(self, throw=True):

        self.navigate_to_main_page()

        Train.TRAINING_TASK = Task.START_WULIN
        self.print(f"Attempting to click Acticity btn")
        try:
            self.instance.click_on_screen(ACTIVITY2_IMAGE)

            self.print(f"Attempting to click Acticity Icon btn")
            self.instance.click_on_screen(
                ACTIVITYICON_IMAGE, IMAGE_FAILED_TIMEOUT)

            self.print(f"Attempting to click Join Acticity btn")
            self.instance.click_on_screen(
                JOINACTIVITY_IMAGE, IMAGE_FAILED_TIMEOUT)

            self.print(f"Attempting to click Start Acticity btn")
            self.instance.click_on_screen(
                STARTACTIVITY_IMAGE, IMAGE_FAILED_TIMEOUT)

            self.print(f"Attempting to click done btn")
            self.instance.click_on_screen(WULIN_DONE_IMAGE, MAX_WULIN_TIMEOUT)

            time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
            self.instance.instance_adb.capture_done_screen()
        except Exception as e:
            self.print(f"Failed to Start Activity: {e}")
            if throw:
                raise e

    def start_mingjiang(self, throw=True):
        try:
            self.navigate_to_main_page(throw)

            Train.TRAINING_TASK = Task.JOIN_TEAM
            self.print("Attemping to click social btn")
            self.instance.click_on_screen(SOCIAL_IMAGE)

            self.print("Attemping to find master")
            self.instance.click_on_screen(MASTER_IMAGE)

            self.print("Attemping to join team")
            self.instance.click_on_screen(JOIN_TEAM_IMAGE)

            time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
            self.instance.instance_adb.capture_done_screen()

            self.navigate_to_main_page(throw)

            Train.TRAINING_TASK = Task.FOLLOW_PLAYER
            self.print("Attemping to find leader icon")
            self.instance.click_on_screen(LEADER_IMAGE)

            self.print("Attemping to follow leader")
            try:
                self.instance.click_on_screen_with_multiple_choice(
                    [MASTER_CONFIRM], [FOLLOW_IMAGE])

                time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
                self.instance.instance_adb.capture_done_screen()
            except Exception as e:
                pass

            self.countdown(MAX_WULIN_TIMEOUT)

            Train.TRAINING_TASK = Task.QUIT_TEAM
            self.print("Attemping to find team btn")
            try:
                self.instance.click_on_screen(TEAM_IMAGE)

                self.print("Attemping to find quit team btn")
                self.instance.click_on_screen(
                    QUIT_TEAM_IMAGE)

            except Exception as e:
                self.print(f"Failed to find team btn: {e}")
                self.print(f"Trying alternate way")

                self.navigate_to_main_page()

                self.instance.instance_adb.adb_tap(25, 175)
                self.instance.instance_adb.capture_screen(25, 175)

                self.print("Attemping to find quit team btn")
                self.instance.click_on_screen_with_multiple_choice(
                    [MASTER_CONFIRM], [QUIT_TEAM2_IMAGE])

            time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
            self.instance.instance_adb.capture_done_screen()

        except Exception as e:
            self.print(f"Failed to start mingjiang: {e}")

    def collect_rewards(self, throw=True):
        try:
            Train.TRAINING_TASK = Task.CLAIM_EXP
            self.print(f"Attempting to collect Exp")
            try:
                if self.instance.click_on_screen_with_multiple_choice(
                        [COLLECT_EXP_IMAGE], [OFFLINE_EXP_TAB_IMAGE], IMAGE_FAILED_TIMEOUT):
                    self.print(f"Offline Exp Collected")
            except Exception as e:
                self.print(f"Failed to collect Exp:{e}")

            time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
            self.instance.instance_adb.capture_done_screen()

            Train.TRAINING_TASK = Task.CLAIM_MONEY
            self.print(f"Attempting to collect money")
            try:
                if self.instance.click_on_screen_with_multiple_choice(
                        [COLLECT_MONEY_IMAGE], [MONEY_COLLECTION_TAB_IMAGE], IMAGE_FAILED_TIMEOUT):
                    self.print(f"Money Collected")
            except Exception as e:
                self.print(f"Failed to collect money:{e}")

            time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
            self.instance.instance_adb.capture_done_screen()

            self.instance.instance_adb.adb_swipe(170, 400, 170, 200)

            Train.TRAINING_TASK = Task.CLAIM_RESOURCES
            self.print(f"Attempting to retrive missed rewards")
            try:
                if self.instance.click_on_screen_with_multiple_choice(
                        [RETRIVE_REWARD_TAB_IMAGE], [MISSED_REWARD_TAB_IMAGE], IMAGE_FAILED_TIMEOUT):
                    self.print(f"Navigated to retrive reward tab")

                count = 0
                while (MISSED_REWARD_MAX_COUNT > count):
                    count += 1
                    if self.instance.click_on_screen_with_multiple_choice(
                            [CONFIRM_IMAGE], [RETRIVE_IMAGE], IMAGE_FAILED_TIMEOUT):
                        continue
                    break
                self.print(f"Rewards Collected")
            except Exception as e:
                self.print(f"Failed to collect reward:{e}")

            time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
            self.instance.instance_adb.capture_done_screen()

            try:
                self.navigate_to_main_page(throw)
            except Exception as e:
                self.print(f"Failed to navigate to main page:{e}")

            Train.TRAINING_TASK = Task.CLAIM_MAIL
            self.print(f"Attempting to retrive rewards in mail")
            self.instance.instance_adb.adb_tap(*OPEN_CHAT_POS)
            self.instance.instance_adb.capture_screen(*OPEN_CHAT_POS)
            try:
                while self.instance.click_on_screen_with_multiple_choice(
                        [COLLECT_ATTACHMENT_IMAGE], [MAIL_CONTANIER_IMAGE, MAILBOX_IMAGE], IMAGE_FAILED_TIMEOUT):
                    self.print(f"Mail Collected")

            except Exception as e:
                self.print(f"Failed to collect mails: {e}")

                time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
                self.instance.instance_adb.capture_done_screen()

                Train.TRAINING_TASK = Task.DELETE_MAIL

                self.print(f"Attempting to delete mails")

                time.sleep(NN_CAPTURE_DONE_IMAGE_DELAY)
                self.instance.instance_adb.capture_done_screen()
                try:
                    if self.instance.click_on_screen_with_multiple_choice(
                            [MASTER_CONFIRM], [DELETE_MAILS_IMAGE], IMAGE_FAILED_TIMEOUT):
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
        print(f"[+]{message}", flush=True, **kwargs)

    def automate_multiple_wulin(self, login_details):

        for login_detail in login_details:
            self.game_switch_account(login_detail, throw=False)
            self.game_login(throw=False)
            # avoid mainpagechacker was found before activity tab shows
            self.countdown(10)
            self.start_wulin(throw=False)
            self.game_logout(throw=False)

    def automatic_multiple_collect_rewards(self, login_details):

        for login_detail in login_details:
            self.game_switch_account(login_detail, throw=False)
            self.game_login(throw=False)
            self.collect_rewards(throw=False)
            self.game_logout(throw=False)

    def automate_multiple_mingjiang(self, login_details):

        for login_detail in login_details:
            self.game_switch_account(login_detail, throw=False)
            self.game_login(throw=False)
            # avoid mainpagechacker was found before activity tab shows
            self.countdown(10)
            self.start_mingjiang(throw=False)
            self.game_logout(throw=False)
