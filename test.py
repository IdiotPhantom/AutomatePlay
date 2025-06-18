from adb import ADB
from setting import *

instance_adb = ADB(ADB_PATH)

instance_adb.adb_get_port()

if not instance_adb.adb_connect():
    raise Exception("Failed to connect port")

print(f"Attempting to key in details")
instance_adb.adb_tap(*ACCOUNT_TEXT_FIELD_POS)
instance_adb.adb_type('KEYCODE_POUND' + "test")
instance_adb.adb_tap(*PASSWORD_TEXT_FIELD_POS)
instance_adb.adb_type("test")