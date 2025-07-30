from pathlib import Path
import serial
import numpy as np
import librosa
import noisereduce as nr
import joblib
import csv
import os
import time
import traceback
from datetime import datetime
from collections import deque
from sklearn.preprocessing import StandardScaler

# === CONFIG ===
SERIAL_PORT     = 'COM3'
BAUD_RATE       = 500000
SAMPLE_RATE     = 16000
BLOCK_SIZE      = 512
WINDOW_DURATION = 2.0
STEP_DURATION   = 1.0
WINDOW_SIZE     = int(SAMPLE_RATE * WINDOW_DURATION)
STEP_SIZE       = int(SAMPLE_RATE * STEP_DURATION)
CSV_FILE        = "log_results.csv"
COMPONENT_NAMES = ['mast','elevator','Gripper','shuttle','envir']

# Feature dims (must match dataset builder)
N_MFCC     = 40
MAX_FRAMES = 63
SR         = SAMPLE_RATE

# === LOAD MODELS & SCALER ===
ocsvm  = joblib.load("svm_oc_model.joblib")
svm    = joblib.load("log_reg_model (1).joblib")
scaler = joblib.load("scaler.joblib")

# === SERIAL INIT ===
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
buffer = deque(maxlen=WINDOW_SIZE)
sample_counter = 0

# === CSV INIT ===
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(["Timestamp","Component","Status","dB","TopFreq1","TopFreq2","TopFreq3"])

# === PREPROCESSING UTILITIES ===
def reduce_noise(sig, sr=SR):
    return nr.reduce_noise(y=sig, sr=sr)

def extract_mfcc(sig, sr=SR, n_mfcc=N_MFCC):
    return librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc)

def pad_mfcc(mfcc, max_frames=MAX_FRAMES):
    n_mfcc, n_frames = mfcc.shape
    if n_frames < max_frames:
        pad = max_frames - n_frames
        return np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    return mfcc[:, :max_frames]

def preprocess_samples(signal):
    # 1) normalize
    raw = signal.astype(np.float32)/np.max(np.abs(signal))
    # 2) denoise
    den = reduce_noise(raw)
    # 3) MFCC
    mf = extract_mfcc(den)
    # 4) pad/truncate
    mf_fixed = pad_mfcc(mf)
    # 5) flatten + scale
    flat = mf_fixed.flatten()[None,:]
    scaled = scaler.transform(flat)
    return scaled, raw

CALIBRATION_OFFSET = 66.0  # Based on SPL meter or calibrator

def compute_db(sig):
    rms = np.sqrt(np.mean(sig.astype(np.float64)**2))
    if rms < 1e-12:
        return 0.0
    dbfs = 20 * np.log10(rms)  # normalized input
    db = dbfs + CALIBRATION_OFFSET
    print(f"[DEBUG] rms={rms:.6f}, dbfs={dbfs:.2f}, db={db:.2f}")
    return db



def compute_top_frequencies(sig, sr=SR, top_n=3):
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig),1/sr)
    mags = np.abs(S)
    idx = np.argsort(mags)[-top_n:][::-1]
    return f[idx]

# === MAIN LOOP ===
while True:
    try:
        # --- SYNC HEADER ---
        b1 = ser.read(1)
        if not b1 or b1[0]!=0xAA:
            continue
        b2 = ser.read(1)
        if not b2 or b2[0]!=0x55:
            continue

        # --- READ BLOCK ---
        raw_bytes = ser.read(BLOCK_SIZE*2)
        if len(raw_bytes) < BLOCK_SIZE*2:
            continue

        # --- BUFFERING ---
        samples = np.frombuffer(raw_bytes,dtype=np.int16)
        buffer.extend(samples)
        sample_counter += len(samples)

        # --- WINDOW READY? ---
        if sample_counter >= STEP_SIZE and len(buffer) >= WINDOW_SIZE:
            sample_counter = 0
            window = np.array(buffer)[-WINDOW_SIZE:]

            # preprocess + inference
            features, float_win = preprocess_samples(window)
            db = compute_db(float_win)
            top_freqs = compute_top_frequencies(float_win)
            scores = ocsvm.decision_function(features)[0]
            print(scores)
            is_normal = 1 if scores >= -0.007414712784148846 else -1
            #is_normal = ocsvm.predict(features)[0]
            label_idx = svm.predict(features)[0]
            label = COMPONENT_NAMES[label_idx]
            status = "NORMAL" if is_normal==1 else "ANOMALY"
            icon = "🟢" if is_normal==1 else "🔴"
            now = datetime.now().isoformat(timespec='seconds')
            freqs_str = [f"{f:.1f}" for f in top_freqs]

            print(f"{icon} {now} - {label} {status} dB={db:.1f}Hz={freqs_str}")

            # log CSV
            with open(CSV_FILE,'a',newline='') as f:
                w = csv.writer(f)
                w.writerow([now,label,status,f"{db:.1f}",*freqs_str])

    except KeyboardInterrupt:
        print("Exiting.")
        break
    except Exception as e:
        print("Runtime error:", e)
        traceback.print_exc()
        time.sleep(1)
