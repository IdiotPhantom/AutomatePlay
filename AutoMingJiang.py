import subprocess
from setting import *
from jianxia import JianXia


def run_automation(instance_name=MASTER_INSTANCE, login_details=LOGIN_DETAILS):
    print(f"Master Instance: {instance_name}")
    try:
        bluestack_instance = JianXia(instance_name)
        bluestack_instance.automate_multiple_mingjiang(login_details)
        print(
            f"Automation on {instance_name} success")
        return True
    except Exception as e:
        print(f"\t[!]Exception {e.__class__.__name__} thrown: {e}")
        print(
            f"Failed to automate on {instance_name}")
        return False


def main():
    # Kill all HD-Player.exe processes
    subprocess.run(["taskkill", "/F", "/IM", "HD-Player.exe"])
    run_automation()


if __name__ == "__main__":
    main()
