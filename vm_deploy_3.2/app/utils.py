# app/utils.py

import time
from threading import Event

def spinner_task(stop_event: Event):
    """
    แสดง spinner ขณะรัน เพื่อเป็น indicator ว่ายังทำงานอยู่
    stop_event: threading.Event ที่ใช้สั่งหยุด loop
    """
    spinner = ['|', '/', '-', '\\']
    idx = 0
    while not stop_event.is_set():
        print(f"\rInitializing fault detection... {spinner[idx]}", end='', flush=True)
        idx = (idx + 1) % len(spinner)
        time.sleep(0.2)
