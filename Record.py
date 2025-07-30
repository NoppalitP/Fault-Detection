from pathlib import Path
import serial
import numpy as np
import wave
import time
from collections import deque
from datetime import datetime

# === CONFIG ===
SERIAL_PORT   = 'COM3'
BAUD_RATE     = 500000
SAMPLE_RATE   = 16000        # Hz
CHANNELS      = 1            # mono
SAMPLE_WIDTH  = 2            # bytes (int16)
WINDOW_SEC    = 2.0          # ความยาวหน้าต่าง (วินาที)
STEP_SEC      = 1.0          # ก้าวหน้าต่าง (วินาที)
OUTPUT_DIR    = Path('recordings')
CHUNK_SIZE    = int(SAMPLE_RATE * STEP_SEC)  # จำนวนตัวอย่างที่อ่านแต่ละครั้ง

# สร้างโฟลเดอร์ถ้ายังไม่มี
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# เรียก buffer สำหรับหน้าต่างเสียง
window_size = int(SAMPLE_RATE * WINDOW_SEC)
buffer = deque(maxlen=window_size)

# เปิด Serial
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print(f"Listening on {SERIAL_PORT} @ {BAUD_RATE} baud...")

# อ่านข้อมูลจาก serial และบันทึกเป็น WAV ด้วย sliding window
try:
    # เติม buffer ครั้งแรก
    required_bytes = window_size * SAMPLE_WIDTH
    collected = bytearray()
    print(f"Collecting initial {WINDOW_SEC}s of data...")
    while len(collected) < required_bytes:
        chunk = ser.read(ser.in_waiting or SAMPLE_WIDTH)
        if chunk:
            collected.extend(chunk)
    init_samples = np.frombuffer(collected[:required_bytes], dtype=np.int16)
    buffer.extend(init_samples.tolist())
    print("Initial buffer filled. Starting sliding window recording...")

    # Loop อ่านและเขียนไฟล์
    while True:
        # อ่าน CHUNK_SIZE ตัวอย่างจาก serial
        bytes_needed = CHUNK_SIZE * SAMPLE_WIDTH
        data = bytearray()
        while len(data) < bytes_needed:
            chunk = ser.read(ser.in_waiting or SAMPLE_WIDTH)
            if chunk:
                data.extend(chunk)
        samples = np.frombuffer(data[:bytes_needed], dtype=np.int16)

        # อัพเดต buffer (auto drop เก่าตาม maxlen)
        buffer.extend(samples.tolist())

        # สร้างไฟล์ WAV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        wav_path = OUTPUT_DIR / f"record_{timestamp}.wav"
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            # เขียนช่วงเวลาปัจจุบัน (WINDOW_SEC)
            arr = np.array(buffer, dtype=np.int16)
            wf.writeframes(arr.tobytes())
        print(f"Saved {wav_path} ({WINDOW_SEC}s window)")

        # รอจนกระทีก้าวถัดไป (ไม่บล็อก อ่านแล้วเกิดไปเรื่อย ๆ)
        # ในที่นี้อ่านทันทีต่อเนื่อง จึงไม่ต้อง sleep

except KeyboardInterrupt:
    print("Stopping recording.")
finally:
    ser.close()
    print("Serial port closed.")
