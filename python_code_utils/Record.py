# -*- coding: utf-8 -*-
"""
This script reads audio data from a serial port, buffers it, and saves it
as overlapping .wav files. It's designed to continuously capture audio
in segments of a specified duration with a defined overlap.
"""
import serial
import numpy as np
import wave
from datetime import datetime
from collections import deque
import time
import os
import sys

# === CONFIGURATION ==========================================================
# --- Serial Port Settings ---
SERIAL_PORT = 'COM5'  # ตั้งค่าพอร์ตอนุกรมที่ต้องการ
BAUD_RATE = 500000     # ตั้งค่า Baud Rate ให้ตรงกับอุปกรณ์

# --- Audio Settings ---
SAMPLE_RATE = 22050    # 16,000 samples per second
CHANNELS = 1           # Mono audio
SAMPLE_WIDTH = 2       # 2 bytes per sample (for 16-bit audio)

# --- Recording Loop Settings ---
BLOCK_SIZE = 512       # จำนวน sample ที่อ่านในแต่ละครั้ง

# --- File Saving Settings ---
WINDOW_DURATION = 30 # ความยาวของเสียงแต่ละไฟล์ (วินาที)
STEP_DURATION = 30.0    # ความถี่ในการบันทึกไฟล์ใหม่ (วินาที)

#============================================================================

OUTPUT_DIRECTORY = r"record"
NAME_COMPONENT= "envir_lab"  # ชื่อของ component ที่บันทึก


# === DERIVED CONSTANTS (DO NOT CHANGE) ======================================
# คำนวณค่าต่างๆ จากการตั้งค่าด้านบน
BLOCK_DURATION_MS = (BLOCK_SIZE / SAMPLE_RATE) * 1000
WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)
STEP_SIZE_SAMPLES = int(SAMPLE_RATE * STEP_DURATION)
# ============================================================================

def create_output_directory(directory_path):
    """
    Checks if the output directory exists and creates it if it doesn't.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"📁 Created directory: {directory_path}")
        except OSError as e:
            print(f"❌ Error creating directory {directory_path}: {e}", file=sys.stderr)
            sys.exit(1)

def save_wave_file(filepath, audio_data, sample_rate, sample_width, channels):
    """
    Saves the provided audio data to a .wav file.

    Args:
        filepath (str): The full path to the output .wav file.
        audio_data (bytes): The raw audio data to save.
        sample_rate (int): The sample rate of the audio.
        sample_width (int): The sample width in bytes (e.g., 2 for 16-bit).
        channels (int): The number of audio channels (e.g., 1 for mono).
    """
    try:
        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
    except Exception as e:
        print(f"❌ Error saving file {filepath}: {e}", file=sys.stderr)

def main():
    """
    Main function to connect to the serial port, read audio data,
    and save it to files.
    """
    # ตรวจสอบและสร้างโฟลเดอร์สำหรับเก็บไฟล์
    create_output_directory(OUTPUT_DIRECTORY)

    ser = None  # Initialize ser to None
    try:
        # --- Connect to Serial Port ---
        print(f"🔌 Connecting to serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=STEP_DURATION)
        print("✅ Connection successful.")
        # Flush any old data in the input buffer
        ser.flushInput()

    except serial.SerialException as e:
        print(f"❌ Critical Error: Could not open serial port {SERIAL_PORT}. {e}", file=sys.stderr)
        print("Please check the port name, permissions, and ensure no other program is using it.")
        sys.exit(1)

    # --- Initialize Buffers and Counters ---
    # ใช้ deque เพื่อเป็น buffer แบบหมุนเวียนที่มีประสิทธิภาพ
    audio_buffer = deque(maxlen=WINDOW_SIZE_SAMPLES)
    new_samples_counter = 0
    file_counter = 0

    print("\n" + "="*50)
    print(f"🎙️  Listening and saving {WINDOW_DURATION}-second .wav files every {STEP_DURATION} second.")
    print(f"    (Overlap = {WINDOW_DURATION - STEP_DURATION} sec, Window size = {WINDOW_SIZE_SAMPLES} samples)")
    print("    Press Ctrl+C to stop.")
    print("="*50 + "\n")

    try:
        while True:
            # 1) หา header 0xAA55
            b1 = ser.read(1)
            if not b1 or b1[0] != 0xAA:
                continue
            b2 = ser.read(1)
            if not b2 or b2[0] != 0x55:
                continue

            # 2) อ่าน PCM data ตามขนาด BLOCK_SIZE*2 = 1024 ไบต์
            packet_bytes = BLOCK_SIZE * SAMPLE_WIDTH  # 512 * 2
            raw = ser.read(packet_bytes)
            if len(raw) < packet_bytes:
                # อ่านไม่ครบ → ข้าม แล้ว sync ใหม่
                continue

            # 3) แปลงเป็น int16 array
            samples = np.frombuffer(raw, dtype=np.int16)

            # 4) เติม buffer และนับ sample
            audio_buffer.extend(samples)
            new_samples_counter += len(samples)

            # 5) ตรวจสอบว่าพร้อมบันทึกไฟล์ใหม่หรือยัง
            if new_samples_counter >= STEP_SIZE_SAMPLES and len(audio_buffer) >= WINDOW_SIZE_SAMPLES:
                new_samples_counter %= STEP_SIZE_SAMPLES

                # ดึงข้อมูลเสียง 1 window เต็ม
                window_data = np.array(audio_buffer)

                # บันทึกเป็น .wav
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{NAME_COMPONENT}_{timestamp}.wav"
                filepath = os.path.join(OUTPUT_DIRECTORY, filename)
                save_wave_file(filepath, window_data.tobytes(),
                            SAMPLE_RATE, SAMPLE_WIDTH, CHANNELS)

                file_counter += 1
                print(f"✅ [{file_counter}] Saved: {filename}")

    except KeyboardInterrupt:
        print("\n🛑 User pressed Ctrl+C. Shutting down gracefully.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        # --- Cleanup ---
        if ser and ser.is_open:
            ser.close()
            print("🔌 Serial port closed.")
        print("👋 Program finished.")


if __name__ == "__main__":
    main()
