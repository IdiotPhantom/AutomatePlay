import os

# bluestack shortcut target: "C:\Program Files\BlueStacks_nxt\HD-Player.exe" --instance Pie64_5 --cmd launchApp --package "com.efun.jxqy.sm" --source desktop_shortcut
BLUESTACK_PATH = r"C:\Program Files\BlueStacks_nxt"
PLAYER_EXE = os.path.join(BLUESTACK_PATH, "HD-Player.exe")
GAME_PACKAGE = "com.efun.jxqy.sm"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADB_PATH = os.path.join(SCRIPT_DIR, "ADB", "adb.exe")
IMAGE_FOLDER_PATH = os.path.join(SCRIPT_DIR, "images")


def join_image_path(filename):
    return os.path.join(IMAGE_FOLDER_PATH, filename)


GAME_ICON_IMAGE = join_image_path("gameicon.png")
LOGIN_IMAGE = join_image_path("login.png")
START_GAME_IMAGE = join_image_path("startgame.png")
EXIT_IMAGE = join_image_path("exit.png")
ACTIVITY_IMAGE = join_image_path("activity.png")
ACTIVITY2_IMAGE = join_image_path("activity2.png")
ACTIVITYICON_IMAGE = join_image_path("activityicon.png")
JOINACTIVITY_IMAGE = join_image_path("joinactivity.png")
STARTACTIVITY_IMAGE = join_image_path("startactivity.png")
OFFLINE_EXP_TAB_IMAGE = join_image_path("offlineexptab.png")
COLLECT_EXP_IMAGE = join_image_path("collectexp.png")
MONEY_COLLECTION_TAB_IMAGE = join_image_path("moneycollectiontab.png")
COLLECT_MONEY_IMAGE = join_image_path("collectmoney.png")
MISSED_REWARD_TAB_IMAGE = join_image_path("missedrewardtab.png")
RETRIVE_REWARD_TAB_IMAGE = join_image_path("retriverewardtab.png")
RETRIVE_IMAGE = join_image_path("retrive.png")
CONFIRM_IMAGE = join_image_path("confirm.png")
MAILBOX_IMAGE = join_image_path("mailbox.png")
MAIL_CONTANIER_IMAGE = join_image_path("mailcontainer.png")
COLLECT_ATTACHMENT_IMAGE = join_image_path("collectattachment.png")
DELETE_MAILS_IMAGE = join_image_path("deletemails.png")
CONFIRM_DELETE_MAILS_IMAGE = join_image_path("confirmdelete.png")
SWITCH_ACCOUNT_IMAGE = join_image_path("switchaccount.png")
SWITCH_ACCOUNT_LOGIN_IMAGE = join_image_path("switchaccountlogin.png")
GOOGLE_LOGIN_IMAGE = join_image_path("googlelogin.png")
GOOGLE_ACC_2_IMAGE = join_image_path("google2.png")
MEMBER_LOGIN_IMAGE = join_image_path("memberlogin.png")
MEMBER_LOGIN_ACCOUNT_IMAGE = join_image_path("memberloginacc.png")
LOGOUT_IMAGE = join_image_path("logout.png")
MAIN_PAGE_CHECKER_IMAGE = join_image_path("mainpagechecker.png")
CHAT_CLOSER_IMAGE = join_image_path("chatcloser.png")
MAIL_CLOSER_IMAGE = join_image_path("mailcloser.png")
ACTIVITY_EXIT_IMAGE = join_image_path("activityexit.png")
INSTANCES = [
    {"name": "Pie64_5"},
    {"name": "Pie64_9"},
    {"name": "Pie64_10"},
    {"name": "Pie64_11"},
    {"name": "Pie64_12"},
    {"name": "Pie64_13"}
]

MASTER_INSTANCE = "Pie64_5"

LOGIN_DETAILS = [
    {"account": "google", "password": GOOGLE_ACC_2_IMAGE},
    {"account": "jiayuliew001", "password": "Liew646488470"},
    {"account": "jiayuliew002", "password": "Liew646488470"},
    {"account": "jyl001", "password": "abc123"},
    {"account": "jyl002", "password": "abc123"},
    {"account": "jyl003", "password": "abc123"}
]

INSTANCE_STARTING_TIME = 15
GAME_START_TIMEOUT = 60
IMAGE_FAILED_TIMEOUT = 5
INTERVAL = 10

MISSED_REWARD_MAX_COUNT = 10
MAX_WULIN_TIMEOUT = 45
BACK_TO_MAIN_PAGE_FAILED_COUNT = 5

OPEN_CHAT_POS = (625, 480)
ACCOUNT_TEXT_FIELD_POS = (450, 180)
PASSWORD_TEXT_FIELD_POS = (450, 250)
SETTING_TAB_POS = (50, 50)
