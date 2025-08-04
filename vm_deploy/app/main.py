from pathlib import Path
import serial
import numpy as np
import librosa
import noisereduce as nr
import joblib
import csv
import time
import traceback
from datetime import datetime, timedelta
from collections import deque
import yaml

# --- โหลด config ---
BASE = Path(__file__).parent.parent     # สมมติโครงสร้าง /deploy/app/main.py
cfg_path = BASE / "app" / "config.yaml"
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

# --- ดึงค่าต่างๆ ---
SERIAL_PORT   = cfg['serial']['port']
BAUD_RATE     = cfg['serial']['baud_rate']
SAMPLE_RATE   = cfg['audio']['sample_rate']
BLOCK_SIZE    = cfg['audio']['block_size']
WINDOW_SIZE   = cfg['window']['size']
STEP_SIZE     = cfg['window']['step']

LOG_DIR       = BASE / cfg['logging']['log_dir']
LOG_DIR.mkdir(exist_ok=True)
ROTATION_MIN  = cfg['logging']['rotation_minutes']

COMPONENTS    = cfg['components']

ocsvm_path    = BASE / cfg['models']['ocsvm']
svm_path      = BASE / cfg['models']['log_reg']
scaler_path   = BASE / cfg['models']['scaler']
MODEL_THRESHOLD = cfg["models"]["threshold"]
TESTER_NAME = cfg["testers"]["name"]

ocsvm = joblib.load(ocsvm_path)
svm = joblib.load(svm_path)
scaler = joblib.load(scaler_path)
# === SERIAL INIT ===
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
buffer = deque(maxlen=WINDOW_SIZE)
sample_counter = 0

# === CSV ROTATION SETUP ===
current_start = datetime.now()
def new_log_file(start_time):
    fname = start_time.strftime(f"{TESTER_NAME}_%Y%m%d_%H%M%S.csv")
    path = LOG_DIR / fname
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Timestamp","Component","Status","dB","TopFreq1","TopFreq2","TopFreq3","Tester_id"])
    return path

current_log = new_log_file(current_start)

# === PREPROCESSING UTILITIES ===
def reduce_noise(sig, sr=SAMPLE_RATE):
    return nr.reduce_noise(y=sig, sr=sr)

def extract_mfcc(sig, sr=SAMPLE_RATE, n_mfcc=40):
    return librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc)

def pad_mfcc(mfcc, max_frames=63):
    n_mfcc, n_frames = mfcc.shape
    if n_frames < max_frames:
        return np.pad(mfcc, ((0,0),(0,max_frames-n_frames)), mode='constant')
    return mfcc[:, :max_frames]

def preprocess_samples(signal):
    scaled = signal.astype(np.float32) / 32768.0
    den    = reduce_noise(scaled)
    mf     = extract_mfcc(den)
    mf_fixed = pad_mfcc(mf)
    flat   = mf_fixed.flatten()[None,:]
    scaled_features = scaler.transform(flat)
    return scaled_features, signal


def compute_db(sig):
    rms = np.sqrt(np.mean((sig.astype(np.float64)/32768.0)**2))
    dBFS = 20*np.log10(rms)
    mic_sensitivity_offset = 94 - (-22)
    est_dbspl = dBFS + mic_sensitivity_offset
    calibrated_offset = 54 - 79
    return est_dbspl + calibrated_offset


def compute_top_frequencies(sig, sr=SAMPLE_RATE, top_n=3):
    S    = np.fft.rfft(sig)
    f    = np.fft.rfftfreq(len(sig),1/sr)
    mags = np.abs(S)
    idx  = np.argsort(mags)[-top_n:][::-1]
    return f[idx]

# === MAIN LOOP ===
try:
    running = True
    while running:
        now = datetime.now()
        if now - current_start >= timedelta(minutes=ROTATION_MIN):
            current_start = now
            current_log = new_log_file(current_start)
            print(f"🔄 Rotated log file: {current_log}")

        try:
            b1 = ser.read(1)
            if not b1 or b1[0] != 0xAA:
                continue
            b2 = ser.read(1)
            if not b2 or b2[0] != 0x55:
                continue

            raw_bytes = ser.read(BLOCK_SIZE*2)
            if len(raw_bytes) < BLOCK_SIZE*2:
                continue

            samples = np.frombuffer(raw_bytes, dtype=np.int16)
            buffer.extend(samples)
            sample_counter += len(samples)

            if sample_counter >= STEP_SIZE and len(buffer) >= WINDOW_SIZE:
                sample_counter = 0
                window = np.array(buffer)[-WINDOW_SIZE:]

                features, float_win = preprocess_samples(window)
                db = compute_db(float_win)
                top_freqs = compute_top_frequencies(float_win)

                score = ocsvm.decision_function(features)[0]
                is_normal = 1 if score >= MODEL_THRESHOLD else -1 
                label_idx = svm.predict(features)[0]
                label = COMPONENTS[label_idx]
                status = "NORMAL" if is_normal==1 else "ANOMALY"

                ts = datetime.now().isoformat(timespec='seconds')
                freqs_str = [f"{f:.1f}" for f in top_freqs]
                print(f"{TESTER_NAME} - {ts} - {label} {status} dB={db:.1f} Hz={freqs_str}")

                with open(current_log, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([ts, label, status, f"{db:.1f}", *freqs_str, TESTER_NAME])

        except Exception as e:
            ts = datetime.now().isoformat(timespec='seconds')
            err_msg = f"[{ts}] Runtime error: {e}"
            print(err_msg)
            traceback.print_exc()

            with open(BASE / "error.log", "a") as logf:
                logf.write(err_msg + "\n")
                logf.write(traceback.format_exc() + "\n")

except KeyboardInterrupt:
    print("🛑 Keyboard interrupt received. Closing...")

finally:
    if ser.is_open:
        ser.close()
    print("✅ Serial port closed.")

