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
AUDIO_DIR = Path(r'C:\Users\Kittisak\Downloads\component_classification_project\python_code\data\component_data_train_test\test')  # โฟลเดอร์ที่เก็บไฟล์เสียง
#AUDIO_DIR = Path(r'data\component_data_train_test\train')  # โฟลเดอร์ที่เก็บไฟล์เสียง

CSV_FILE = 'log_results_test.csv'
SAMPLE_RATE = 16000                  # อัตราสุ่มเป้าหมาย
WINDOW_DURATION = 2.0                # หน้าต่างประมวลผล (วินาที)
STEP_DURATION = 1.0                  # ก้าวการประมวลผล (วินาที)
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
STEP_SIZE = int(SAMPLE_RATE * STEP_DURATION)
N_MFCC = 40
COMPONENT_NAMES = ['mast', 'elevator', 'Gripper', 'shuttle', 'envir']

# === LOAD MODELS ===
try:
    ocsvm = joblib.load('svm_oc_model.joblib')
    svm = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
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
def reduce_noise_on_segment(segment, sr=SAMPLE_RATE):
    """
    Apply noise reduction on a single audio segment using an estimated noise profile.
    """
    return nr.reduce_noise(y=segment, sr=sr)


def extract_mfcc(segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extracts MFCC features from an audio segment.
    """
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def pad_mfcc(mfcc, max_frames):
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > max_frames:
        mfcc = mfcc[:, :max_frames]
    return mfcc


def compute_db(signal):
    rms = np.sqrt(np.mean(np.square(signal)))
    return 20 * np.log10(rms + 1e-6) + 95.0


def compute_top_frequencies(signal, sr=SAMPLE_RATE, top_n=3):
    spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(len(signal), d=1.0/sr)
    magnitude = np.abs(spectrum)
    top_indices = np.argsort(magnitude)[-top_n:][::-1]
    return freq[top_indices]


def preprocess_segment(segment, max_frames, sr=SAMPLE_RATE):
    """
    Full preprocessing pipeline for a windowed segment:
     1. Normalize
     2. Noise reduction
     3. MFCC extraction
     4. Padding/truncation
     5. Flatten + scaling
    Returns:
        features (np.ndarray), float_segment (np.ndarray)
    """
    try:
        # normalize int16/float32
        float_seg = segment.astype(np.float32)
        if float_seg.max() > 1.0:
            float_seg /= 32768.0

        # noise reduction
        reduced = reduce_noise_on_segment(float_seg, sr)

        # MFCC
        mfcc = extract_mfcc(reduced, sr, N_MFCC)
        padded = pad_mfcc(mfcc, max_frames)
        flat = padded.reshape(1, -1)

        # scale
        scaled = scaler.transform(flat)
        return scaled, reduced
    except Exception as e:
        print('❌ Preprocessing error:', e)
        traceback.print_exc()
        return None, None

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

        features, proc_seg =                                                                                                                 preprocess_segment(segment, max_frames, sr)
        if features is None:
            continue

        db_value = compute_db(proc_seg)
        freqs = compute_top_frequencies(proc_seg, sr)
        freqs_str = [f'{f:.1f}' for f in freqs]
        scores = ocsvm.decision_function(features)[0]
        print(scores)
        is_norm = 1 if scores >= -0.007414712784148846 else -1
        #is_norm = ocsvm.predict(features)[0]
        label_idx = svm.predict(features)[0]
        label = COMPONENT_NAMES[label_idx] if 0 <= label_idx < len(COMPONENT_NAMES) else f'Unknown({label_idx})'
        status = 'NORMAL' if is_norm == 1 else 'ANOMALY'
        icon = '🟢' if is_norm == 1 else '🔴'
        timestamp = datetime.now().isoformat(timespec='seconds')

        print(f"{icon} {timestamp} - File: {file_path.name}, Component: {label}, Status: {status}, dB: {db_value:.1f}, Top Freqs: {freqs_str}")

        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, file_path.name, label, status, f'{db_value:.1f}', *freqs_str])

    print(f'✅ Completed processing {file_path.name}\n')