import numpy as np
import librosa
import time
from datetime import datetime
defaultdict = __import__('collections').defaultdict
from collections import deque, defaultdict
import joblib
import traceback
import noisereduce as nr
import csv
import os
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# === CONFIG ===
AUDIO_DIR = Path(r'record\anormal')  # โฟลเดอร์ที่เก็บไฟล์เสียง
#AUDIO_DIR = Path(r'data\component_data_train_test\train')  # โฟลเดอร์ที่เก็บไฟล์เสียง

CSV_FILE = 'log_results_test.csv'
SAMPLE_RATE = 22050                  # อัตราสุ่มเป้าหมาย
WINDOW_DURATION = 2.0                # หน้าต่างประมวลผล (วินาที)
STEP_DURATION = 1.0                  # ก้าวการประมวลผล (วินาที)
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
STEP_SIZE = int(SAMPLE_RATE * STEP_DURATION)
N_MFCC = 40
COMPONENT_NAMES = ['mast', 'elevator', 'Gripper', 'shuttle', 'envir']
SR = 22050
# === LOAD MODELS ===
try:
    ocsvm = joblib.load('model_file\svm_oc_model.joblib')
    svm = joblib.load('model_file\svm_model.joblib')
    scaler = joblib.load('model_file\scaler.joblib')
    print('✅ Models loaded')
except Exception as e:
    print('❌ Model loading error:', e)
    raise

# === INIT CSV LOG ===
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'File', 'Component', 'Status', 'dB', 'TopFreq1', 'TopFreq2', 'TopFreq3'])

# === UTILS ===
def reduce_noise(sig, sr=SR):
    return nr.reduce_noise(y=sig, sr=sr)


def extract_mfcc(segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extracts MFCC features from an audio segment.
    """
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def pad_mfcc(mfcc, max_frames = 63):
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > max_frames:
        mfcc = mfcc[:, :max_frames]
    return mfcc


def preprocess_samples(signal):
    # 1) normalize
    raw = signal.astype(np.float32)
    # 2) denoise
    den = reduce_noise(raw)
    # 3) MFCC
    mf = extract_mfcc(den)
    # 4) pad/truncate
    mf_fixed = pad_mfcc(mf)
    # 5) flatten + scale
    flat = mf_fixed.flatten()[None,:]
    scaled = scaler.transform(flat)
    return  signal



def compute_db(sig):
    rms = np.sqrt(np.mean(signal**2))
    print(f"RMS: {rms}")
    dBFS = 20 * np.log10(rms) 
    mic_sensitivity_offset = 94 - (-22)
    estimated_dbspl = dBFS + mic_sensitivity_offset
    calibrated_dbspl_offset = 54 - 80 + 68 - 64+2
    dBSPL = estimated_dbspl + calibrated_dbspl_offset 
    return dBSPL



def compute_top_frequencies(sig, sr=SR, top_n=3):
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig),1/sr)
    mags = np.abs(S)
    idx = np.argsort(mags)[-top_n:][::-1]
    return f[idx]

# === PROCESS EACH FILE ===
for file_path in AUDIO_DIR.iterdir():
    if not file_path.suffix.lower() in ('.wav', '.mp3', '.flac'):
        continue
    print(f'🎧 Processing file: {file_path.name}')

    # load full signal
    signal, sr = librosa.load(str(file_path), sr=SAMPLE_RATE)
    max_frames_est = int(WINDOW_SIZE / (WINDOW_SIZE / N_MFCC / 2))  # fallback estimate
    # determine max frames from first window
    first = signal[:WINDOW_SIZE]
    mfcc_first = extract_mfcc(first, sr, N_MFCC)
    max_frames = mfcc_first.shape[1]

    # sliding window
    for i in range(0, len(signal), STEP_SIZE):
        segment = signal[i:i+WINDOW_SIZE]
        if len(segment) < WINDOW_SIZE:
            break

        proc_seg =   preprocess_samples(signal)
        
        # if features is None:
        #     continue

        db_value = compute_db(segment)
        print()
        print(db_value)

        # freqs = compute_top_frequencies(proc_seg, sr)
        # freqs_str = [f'{f:.1f}' for f in freqs]
        # scores = ocsvm.decision_function(features)[0]
        # print(scores)
        # is_norm = 1 if scores >= -0.007414712784148846 else -1
        # #is_norm = ocsvm.predict(features)[0]
        # label_idx = svm.predict(features)[0]
        # label = COMPONENT_NAMES[label_idx] if 0 <= label_idx < len(COMPONENT_NAMES) else f'Unknown({label_idx})'
        # status = 'NORMAL' if is_norm == 1 else 'ANOMALY'
        # icon = '🟢' if is_norm == 1 else '🔴'
        # timestamp = datetime.now().isoformat(timespec='seconds')

        # print(f"{icon} {timestamp} - File: {file_path.name}, Component: {label}, Status: {status}, dB: {db_value:.1f}, Top Freqs: {freqs_str}")

        # with open(CSV_FILE, mode='a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([timestamp, file_path.name, label, status, f'{db_value:.1f}', *freqs_str])

    print(f'✅ Completed processing {file_path.name}\n')