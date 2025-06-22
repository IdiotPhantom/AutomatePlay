import os
import time
from enum import Enum

# --- Base Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLUESTACK_PATH = r"C:\Program Files\BlueStacks_nxt"
ADB_PATH = os.path.join(SCRIPT_DIR, "ADB", "adb.exe")
PLAYER_EXE = os.path.join(BLUESTACK_PATH, "HD-Player.exe")
IMAGE_FOLDER_PATH = os.path.join(SCRIPT_DIR, "images")
NN_FILE_NAME = "model.pth"
SCREEN_IMAGE = os.path.join(SCRIPT_DIR, "screen.png")

# --- Game Configuration ---
GAME_PACKAGE = "com.efun.jxqy.sm"
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540

# --- BlueStacks Instances ---
INSTANCES = [{"name": f"Pie64_{i}"} for i in range(5, 14)]
MASTER_INSTANCE = "Pie64_5"

# --- Login Details (Use caution with storing credentials) ---
LOGIN_DETAILS = [
    {"account": "google", "password": "images/google2.png"},
    {"account": "jiayuliew001", "password": "Liew646488470"},
    {"account": "jiayuliew002", "password": "Liew646488470"},
    {"account": "jyl001", "password": "abc123"},
    {"account": "jyl002", "password": "abc123"},
    {"account": "jyl003", "password": "abc123"}
]

# --- Timing Settings ---
INSTANCE_STARTING_TIME = 10
GAME_START_TIMEOUT = 60
IMAGE_FAILED_TIMEOUT = 5
INTERVAL = 10
MISSED_REWARD_MAX_COUNT = 15
MAX_WULIN_TIMEOUT = 45
BACK_TO_MAIN_PAGE_FAILED_COUNT = 5
NN_CAPTURE_DONE_IMAGE_DELAY = 5

# --- Screen Coordinates ---
OPEN_CHAT_POS = (625, 480)
ACCOUNT_TEXT_FIELD_POS = (450, 180)
PASSWORD_TEXT_FIELD_POS = (450, 250)
SETTING_TAB_POS = (50, 50)
TEAM_SETTING_POS = (25, 175)


CONFIDENCE_THRESHOLD = 0.7  # adjust as needed

# --- Utility ---


def join_image_path(filename):
    return os.path.join(IMAGE_FOLDER_PATH, filename)


SCREEN_IMAGE = os.path.join(SCRIPT_DIR, "screen.png")
# --- Image Assets ---
GAME_ICON_IMAGE = join_image_path("gameicon.png")
LOGIN_IMAGE = join_image_path("login.png")
START_GAME_IMAGE = join_image_path("startgame.png")
EXIT_IMAGE = join_image_path("exit.png")

# Activity-related
ACTIVITY_IMAGE = join_image_path("activity.png")
ACTIVITY2_IMAGE = join_image_path("activity2.png")
ACTIVITYICON_IMAGE = join_image_path("activityicon.png")
ACTIVITY_EXIT_IMAGE = join_image_path("activityexit.png")
JOINACTIVITY_IMAGE = join_image_path("joinactivity.png")
STARTACTIVITY_IMAGE = join_image_path("startactivity.png")

# Rewards
OFFLINE_EXP_TAB_IMAGE = join_image_path("offlineexptab.png")
COLLECT_EXP_IMAGE = join_image_path("collectexp.png")
MONEY_COLLECTION_TAB_IMAGE = join_image_path("moneycollectiontab.png")
COLLECT_MONEY_IMAGE = join_image_path("collectmoney.png")
MISSED_REWARD_TAB_IMAGE = join_image_path("missedrewardtab.png")
RETRIVE_REWARD_TAB_IMAGE = join_image_path("retriverewardtab.png")
RETRIVE_IMAGE = join_image_path("retrive.png")
CONFIRM_IMAGE = join_image_path("confirm.png")

# Mail
MAILBOX_IMAGE = join_image_path("mailbox.png")
MAIL_CONTANIER_IMAGE = join_image_path("mailcontainer.png")
COLLECT_ATTACHMENT_IMAGE = join_image_path("collectattachment.png")
DELETE_MAILS_IMAGE = join_image_path("deletemails.png")
MAIL_CLOSER_IMAGE = join_image_path("mailcloser.png")

# Login
SWITCH_ACCOUNT_IMAGE = join_image_path("switchaccount.png")
SWITCH_ACCOUNT_LOGIN_IMAGE = join_image_path("switchaccountlogin.png")
GOOGLE_LOGIN_IMAGE = join_image_path("googlelogin.png")
GOOGLE_ACC_2_IMAGE = join_image_path("google2.png")
MEMBER_LOGIN_IMAGE = join_image_path("memberlogin.png")
MEMBER_LOGIN_ACCOUNT_IMAGE = join_image_path("memberloginacc.png")

# General UI
LOGOUT_IMAGE = join_image_path("logout.png")
MAIN_PAGE_CHECKER_IMAGE = join_image_path("mainpagechecker.png")
CHAT_CLOSER_IMAGE = join_image_path("chatcloser.png")
MASTER_CONFIRM = join_image_path("masterconfirm.png")

EXIT_STORE_IMAGE = join_image_path("exitstore.png")

# Wulin / Team System
WULIN_DONE_IMAGE = join_image_path("wulindone.png")
SOCIAL_IMAGE = join_image_path("social.png")
MASTER_IMAGE = join_image_path("master.png")
JOIN_TEAM_IMAGE = join_image_path("jointeam.png")
LEADER_IMAGE = join_image_path("leader.png")
FOLLOW_IMAGE = join_image_path("follow.png")
QUIT_TEAM_IMAGE = join_image_path("quitteam.png")
QUIT_TEAM2_IMAGE = join_image_path("quitteam2.png")
TEAM_IMAGE = join_image_path("team.png")


class Train:
    TRAINING_TASK = None
    TRAINING = True


# Get current local time as a struct_time
current_time = time.localtime()

# Format it as a date string (e.g., YYYY-MM-DD)
date_string = time.strftime("%Y-%m-%d", current_time)
TRAINING_DATA_PATH = os.path.join(SCRIPT_DIR, "data")
TRAINING_DATA_NAME = os.path.join(TRAINING_DATA_PATH, f"data{date_string}.csv")

TRAINING_IMAGE_PATH = os.path.join(SCRIPT_DIR, "training_images")

AUGMENTED_DATA_PATH = os.path.join(SCRIPT_DIR, "augmented_data")
AUGMENTED_DATA_NAME = os.path.join(AUGMENTED_DATA_PATH, f"augmented_data.csv")

AUGMENTED_IMAGE_PATH = os.path.join(SCRIPT_DIR, "augmented_images")

FEEDBACK_DATA_PATH = os.path.join(SCRIPT_DIR, "feedback")
FEEDBACK_DATA_FILE = os.path.join(FEEDBACK_DATA_PATH, "feedback.csv")

LEARNING_IMAGE_PATH = os.path.join(SCRIPT_DIR, "learning_images")

# --- Task Enum ---
class Task(Enum):
    LOGIN = 0
    SWITCH_ACCOUNT = 1
    BACK_TO_MAIN_PAGE = 2
    COLLECT_REWARDS = 3
    START_WULIN = 4
    CLAIM_MAIL = 5
    CLAIM_EXP = 6
    LOGOUT = 7
    FOLLOW_PLAYER = 8
    JOIN_TEAM = 9
    QUIT_TEAM = 10
    CLAIM_MONEY = 11
    CLAIM_RESOURCES = 12
    DELETE_MAIL = 13
    # Add more tasks as needed
