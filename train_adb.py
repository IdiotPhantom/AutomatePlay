from setting import ADB_PATH, SCREEN_IMAGE
import subprocess
import psutil
import re
import os
import cv2


class TrainADB:
    def __init__(self, path=ADB_PATH, device_ip="127.0.0.1", device_port=None):
        self.path = path
        self.device_ip = device_ip
        self.device_port = device_port

    def adb_get_port(self):
        self.device_port = self.get_bluestacks_adb_port()
        if self.device_port == None:
            return False
        return True

    def get_bluestacks_adb_port(self):
        # Try using `adb devices` first
        try:
            result = subprocess.run(
                [self.path, "devices"],
                capture_output=True, text=True
            )
            lines = result.stdout.strip().splitlines()[1:]  # Skip header

            for line in lines:
                if "127.0.0.1" in line and "device" in line:
                    match = re.search(r"127\.0\.0\.1:(\d+)", line)
                    if match:
                        port = int(match.group(1))
                        self.print(
                            f"[ADB] Port found from adb devices: {port}")
                        return port
        except Exception as e:
            self.print(f"[ADB] Error running adb devices: {e}")

        # Fallback: use psutil
        for conn in psutil.net_connections(kind='tcp'):
            if conn.laddr and conn.laddr.ip == '127.0.0.1' and conn.status == 'LISTEN':
                pid = conn.pid
                port = conn.laddr.port
                if pid:
                    try:
                        proc = psutil.Process(pid)
                        if proc.name().lower() == "hd-player.exe":
                            self.print(
                                f"[psutil] Port found from process: {port}")
                            return port
                    except psutil.NoSuchProcess:
                        pass

        self.print("❌ No ADB port found.")
        return None

    def adb_connect(self):
        if self.device_port is None:
            self.print("Device port not set, cannot connect.")
            return False
        cmd = f'"{self.path}" connect {self.device_ip}:{self.device_port}'
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        self.print(f"adb connect output: {output}")

        if "connected to" in output.lower() or "already connected" in output.lower():
            self.print(
                f"Successfully connected to {self.device_ip}:{self.device_port}")
            return True
        else:
            self.print(
                f"Failed to connect to {self.device_ip}:{self.device_port}")
            return False

    def adb_tap(self, x, y):
        cmd = f'"{self.path}" -s {self.device_ip}:{self.device_port} shell input tap {x} {y}'
        subprocess.run(cmd, shell=True)

    def adb_swipe(self, x1, y1, x2, y2, duration=300):
        cmd = f'"{self.path}" -s {self.device_ip}:{self.device_port} shell input swipe {x1} {y1} {x2} {y2} {duration}'
        subprocess.run(cmd, shell=True)

    def adb_type(self, text, clear_before=True, clear_length=30):
        if clear_before:
            cmd_clear = f'"{self.path}" -s {self.device_ip}:{self.device_port} shell ' + \
                ' '.join(['input keyevent 67' for _ in range(clear_length)])
            subprocess.run(cmd_clear, shell=True)

        safe_text = text.replace(' ', '%s')
        cmd = f'"{self.path}" -s {self.device_ip}:{self.device_port} shell input text {safe_text}'
        subprocess.run(cmd, shell=True)

    def take_screenshot(self):
        subprocess.run(
            f'"{self.path}" -s {self.device_ip}:{self.device_port} shell screencap -p /sdcard/screen.png',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        subprocess.run(
            f'"{self.path}" -s {self.device_ip}:{self.device_port} pull /sdcard/screen.png screen.png',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def tap_button_on_screen(self, image, tap=True):
        self.take_screenshot()

        screen = cv2.imread(SCREEN_IMAGE, cv2.IMREAD_GRAYSCALE)
        button = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        if screen is None or button is None:
            raise ValueError(
                "One of the images failed to load. Check file paths and formats.")

        result = cv2.matchTemplate(screen, button, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        threshold = 0.8
        if max_val >= threshold:
            filename = os.path.basename(image)
            top_left = max_loc
            h, w = button.shape[:2]
            center_x = top_left[0] + w // 2
            center_y = top_left[1] + h // 2
            if tap:
                self.adb_tap(center_x, center_y)
            self.print(f"{filename} found at : ({center_x}, {center_y})")
            return {"x": center_x, "y": center_y}
        else:
            return None

    def print(self, message, **kwargs):
        print(f"\t[*]{message}", flush=True, **kwargs)
