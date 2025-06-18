from adb import ADB
from setting import *

instance_ADB = ADB(ADB_PATH)

instance_ADB.adb_get_port()

if not instance_ADB.adb_connect():
    raise Exception("Failed to connect port")

instance_ADB.take_screenshot()