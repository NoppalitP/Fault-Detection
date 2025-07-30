import serial
import numpy as np
import time
import mysql.connector
from datetime import datetime

# CONFIG
PORT = 'COM11'
BAUD = 921600
BUFFER_SIZE = 256  # จุดที่จะใช้สำหรับ FFT
SAMPLING_RATE = 955  # Hz (จากการวัดจริง)

MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '170970022Za-',
    'database': 'gyro'
}
TABLE_NAME = 'fft_result'

# สร้างตาราง
def create_table(cursor):
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME(3),
            sampling_rate FLOAT,
            dominant_freq FLOAT,
            max_magnitude FLOAT
        )
    ''')

# คำนวณ FFT
def compute_fft(signal, rate):
    N = len(signal)
    signal = np.array(signal) - np.mean(signal)  # remove DC bias
    freqs = np.fft.rfftfreq(N, d=1.0 / rate)
    fft_vals = np.fft.rfft(signal)
    magnitudes = np.abs(fft_vals)

    dominant_freq = freqs[np.argmax(magnitudes)]
    max_mag = np.max(magnitudes)

    return dominant_freq, max_mag

# MAIN
ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"📡 Listening on {PORT}...")

conn = mysql.connector.connect(**MYSQL_CONFIG)
cursor = conn.cursor()
create_table(cursor)

buffer = []

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        try:
            ax, ay, az = map(float, line.split(','))

            # เลือกแกนใดแกนหนึ่ง เช่น z-axis
            buffer.append(az)

            if len(buffer) >= BUFFER_SIZE:
                # ทำ FFT
                dominant_freq, max_mag = compute_fft(buffer, SAMPLING_RATE)
                timestamp = datetime.now()

                print(f"[FFT] Dominant: {dominant_freq:.2f} Hz | Max Mag: {max_mag:.2f}")

                # ส่งเข้า MySQL
                cursor.execute(f'''
                    INSERT INTO {TABLE_NAME} (timestamp, sampling_rate, dominant_freq, max_magnitude)
                    VALUES (%s, %s, %s, %s)
                ''', (timestamp, SAMPLING_RATE, dominant_freq, max_mag))
                conn.commit()

                # ล้าง buffer
                buffer = []

        except ValueError:
            print(f"[WARN] Invalid: {line}")
except KeyboardInterrupt:
    print("🛑 Stopped.")
finally:
    ser.close()
    conn.close()
