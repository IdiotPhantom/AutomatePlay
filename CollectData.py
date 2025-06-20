from setting import *
from jianxia import JianXia


def run_automation(instance_name=MASTER_INSTANCE, login_details=LOGIN_DETAILS):
    print(f"Master Instance: {instance_name}")
    try:
        bluestack_instance = JianXia(instance_name, terminate_on_destroy=False)
        bluestack_instance.automate_multiple_wulin(login_details)
        bluestack_instance.automatic_multiple_collect_rewards(login_details)
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
    Train.TRAINING = True
    run_automation()
    Train.TRAINING = True


if __name__ == "__main__":
    main()
