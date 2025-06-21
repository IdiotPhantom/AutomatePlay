from train_jianxia import TrainJianXia
from setting import MASTER_INSTANCE, LOGIN_DETAILS, INSTANCE_STARTING_TIME
import subprocess
import traceback


def run_automation(instance_name=MASTER_INSTANCE, login_details=LOGIN_DETAILS):
    print(f"Master Instance: {instance_name}")

    if is_process_running("HD-Player.exe"):
        starting_time = 0
    else:
        starting_time = INSTANCE_STARTING_TIME

    try:
        bluestack_instance = TrainJianXia(
            instance_name, terminate_on_destroy=False, starting_time=starting_time)

        for login_detail in login_details:
            bluestack_instance.game_switch_account(login_detail)
            bluestack_instance.game_login()
            bluestack_instance.game_logout()

        print(
            f"Automation on {instance_name} success")
        return True
    except Exception as e:
        print(f"[!]Exception {e.__class__.__name__} thrown: {e}")
        print(
            f"Failed to automate on {instance_name}")
        traceback.print_exc()
        return False


def is_process_running(process_name):
    result = subprocess.run(["tasklist"], stdout=subprocess.PIPE, text=True)
    return process_name.lower() in result.stdout.lower()


def main():
    run_automation()


if __name__ == "__main__":
    main()
