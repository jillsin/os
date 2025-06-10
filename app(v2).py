import psutil
import time
import datetime


def get_system_status():
    return {
        # "timestamp": datetime.datetime.now(),
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "battery_percent": (
            psutil.sensors_battery().percent if psutil.sensors_battery() else 100
        ),
        "plugged": (
            int(psutil.sensors_battery().power_plugged)
            if psutil.sensors_battery()
            else 0
        ),
        # "idle_time_sec": get_idle_time(),
        "hour": datetime.datetime.now().hour,
        "cpu_freq_mhz": psutil.cpu_freq().current,
        "net_sent_rate": psutil.net_io_counters().bytes_sent,
        "net_recv_rate": psutil.net_io_counters().bytes_recv,
    }


import joblib
import pandas as pd

model = joblib.load("model_rf_tuned.pkl")  # mô hình đã lưu


def predict_low_activity(features: dict):
    df = pd.DataFrame([features])
    y_pred = model.predict(df)
    return y_pred[0]


import os


def perform_power_action(predicted_label):
    if predicted_label == 1:
        print("Low activity detected. Putting system to sleep...")
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")  # Windows
    else:
        print("System active. No action taken.")


while True:
    features = get_system_status()
    label = predict_low_activity(features)
    perform_power_action(label)
    time.sleep(30)  # chạy mỗi 5 phút
