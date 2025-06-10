import os
import time
import datetime
import psutil
import subprocess
import pandas as pd
import joblib

# Tải mô hình đã huấn luyện
model = joblib.load("model.pkl")  # 🔁 Đổi path cho đúng


def get_idle_time():
    try:
        idle_ms = int(subprocess.check_output("xprintidle").decode())
        return idle_ms / 1000  # giây
    except Exception as e:
        print(f"Error getting idle time: {e}")
        return 0


def get_system_status():
    battery = psutil.sensors_battery()
    return {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "battery_percent": battery.percent if battery else 100,
        "plugged": int(battery.power_plugged) if battery else 0,
        "idle_time_sec": get_idle_time(),
        "hour": datetime.datetime.now().hour,
        "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        "net_sent_rate": psutil.net_io_counters().bytes_sent,
        "net_recv_rate": psutil.net_io_counters().bytes_recv,
    }


def predict_low_activity(features):
    df = pd.DataFrame([features])
    return model.predict(df)[0]


def apply_power_saving():
    print(f"[{datetime.datetime.now()}] 🔋 Low activity → Kích hoạt tiết kiệm điện")
    os.system("xrandr --output eDP-1 --brightness 0.5")
    os.system("xset dpms 60 120 180")
    os.system("sudo cpufreq-set -g powersave")


def restore_power_settings():
    print(f"[{datetime.datetime.now()}] ⚡ Active → Khôi phục bình thường")
    os.system("xrandr --output eDP-1 --brightness 1.0")
    os.system("xset dpms 300 600 900")
    os.system("sudo cpufreq-set -g powersave")


if __name__ == "__main__":
    last_state = None
    while True:
        features = get_system_status()
        label = predict_low_activity(features)

        if label == 0 and last_state != 0:
            apply_power_saving()
        elif label == 1 and last_state != 1:
            restore_power_settings()

        last_state = label
        time.sleep(30)  # mỗi 5 phút kiểm tra
