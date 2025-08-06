import csv
import time
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import librosa
import numpy as np
import noisereduce as nr
import serial
import yaml
import wave
import sys
from collections import deque
import threading

def spinner_task():
    spinner = ['|', '/', '-', '\\']
    idx = 0
    while True:
        print(f"\rInitializing fault detection... {spinner[idx]}", end='', flush=True)
        idx = (idx + 1) % len(spinner)
        time.sleep(0.2)


def setup_logging(log_file: Path):
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )


def new_log_file(start_time: datetime, log_dir: Path, tester_name: str) -> Path:
    fname = start_time.strftime(f"{tester_name}_%Y%m%d_%H%M%S.csv")
    path = log_dir / fname
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Component", "Status",
            "dB", "TopFreq1", "TopFreq2", "TopFreq3", "Tester_id"
        ])
    return path


def save_wave_file(filepath: str, audio_data: bytes, sample_rate: int, sample_width: int):
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
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
    except Exception as e:
        print(f"❌ Error saving file {filepath}: {e}", file=sys.stderr)


def reduce_noise(sig: np.ndarray, sr: int) -> np.ndarray:
    return nr.reduce_noise(y=sig, sr=sr)


def extract_mfcc(sig: np.ndarray, sr: int, n_mfcc: int = 40, hop_length: int = 512) -> np.ndarray:
    return librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)


def pad_mfcc(mfcc: np.ndarray, max_frames: int) -> np.ndarray:
    n_mfcc, n_frames = mfcc.shape
    if n_frames < max_frames:
        return np.pad(mfcc, ((0, 0), (0, max_frames - n_frames)), mode='constant')
    return mfcc[:, :max_frames]


def preprocess_file(wav_path: Path, scaler, sample_rate: int, n_mfcc: int, max_frames: int, hop_length: int):
    sig, sr = librosa.load(str(wav_path), sr=sample_rate)
    denoised = reduce_noise(sig, sr=sr)
    mf = extract_mfcc(denoised, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mf_fixed = pad_mfcc(mf, max_frames=max_frames)
    flat = mf_fixed.flatten()[None, :]
    features = scaler.transform(flat)
    return features, sig


def compute_db(sig: np.ndarray) -> float:
    rms = librosa.feature.rms(y=sig)[0]  # shape (T,)
    # Convert to dBFS (max=0 dB)
    dbfs = librosa.amplitude_to_db(rms, ref=1)
    # Apply microphone calibration offset to estimate dB SPL
    mic_offset = 94 - (-22) +54 - 82 # example: device outputs -22 dBFS at 94 dB SPL
    dbspl = dbfs + mic_offset
    # Return max SPL in window
    return float(np.max(dbspl))


def compute_top_frequencies(sig: np.ndarray, sr: int, top_n: int = 3) -> np.ndarray:
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1/sr)
    mags = np.abs(S)
    idx = np.argsort(mags)[-top_n:][::-1]
    return f[idx]


def batch_predict(wav_dir: Path, log_path: Path, scaler, ocsvm, svm, components,
                  model_threshold, sample_rate, n_mfcc, max_frames, hop_length, tester_name, ts_array):
    for index, wav_file in enumerate(sorted(wav_dir.glob("window_*.wav"))):
        try:
            features, denoised = preprocess_file(
                wav_file, scaler, sample_rate,
                n_mfcc, max_frames, hop_length
            )
            db = compute_db(denoised)
            top_freqs = compute_top_frequencies(denoised, sr=sample_rate)
            score = ocsvm.decision_function(features)[0]
            is_normal = 1 if score >= model_threshold else -1
            label_idx = svm.predict(features)[0]
            label = components[label_idx]
            status = "NORMAL" if is_normal == 1 else "ANOMALY"

            freqs_str = [f"{f:.1f}" for f in top_freqs]
            row = [ts_array[index], label, status, f"{db:.1f}", *freqs_str, tester_name]

            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            logging.info(f"Batch {wav_file.name}: {label} {status} dB={db:.1f} Hz={freqs_str}")
        except Exception as e:
            logging.error(f"Error processing {wav_file}", exc_info=e)
    for wav_file in wav_dir.glob("window_*.wav"):
        wav_file.unlink()


def main():
    base = Path(__file__).resolve().parent.parent
    with open(base / "app" / "config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    serial_port = cfg['serial']['port']
    baud_rate = cfg['serial']['baud_rate']
    sample_rate = cfg['audio']['sample_rate']
    block_size = cfg['audio']['block_size']
    window_size = cfg['window']['size']
    step_size = cfg['window']['step']
    batch_size = cfg['batch']['size']
    tester_name = cfg['testers']['name']
    components = cfg['components']
    model_threshold = cfg['models']['threshold']
    n_mfcc = cfg['audio'].get('n_mfcc', 40)
    hop_length = cfg['audio'].get('hop_length', 512)

    scaler = joblib.load(base / cfg['models']['scaler'])
    expected_dim = scaler.mean_.shape[0]
    max_frames = expected_dim // n_mfcc
    ocsvm = joblib.load(base / cfg['models']['ocsvm'])
    svm = joblib.load(base / cfg['models']['log_reg'])

    log_dir = base / cfg['logging']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = base / cfg['batch']['wav_dir']
    wav_dir.mkdir(parents=True, exist_ok=True)

    app_log = base / "app.log"
    setup_logging(app_log)
    logging.info("Starting batch fault detection monitoring")

    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        logging.info("Serial port opened successfully")
    except Exception as se:
        logging.error("Unable to open serial port", exc_info=se)
        return

    buffer = deque(maxlen=window_size)
    sample_counter = batch_counter = file_log_batch_counter = 0
    current_start = datetime.now()
    ts_array = []
    current_log = new_log_file(current_start, log_dir, tester_name)
    logging.info(f"Rotated log file: {current_log}")

    spinner_thread = threading.Thread(target=spinner_task, daemon=True)
    spinner_thread.start()
    try:
        while True:
            now = datetime.now()
            if file_log_batch_counter >= 180:
                current_start = now
                ts_array = []
                current_log = new_log_file(current_start, log_dir, tester_name)
                logging.info(f"Rotated log file: {current_log}")
                file_log_batch_counter = 0

            b1 = ser.read(1)
            if not b1 or b1[0] != 0xAA:
                continue
            b2 = ser.read(1)
            if not b2 or b2[0] != 0x55:
                continue

            raw_bytes = ser.read(block_size * 2)
            if len(raw_bytes) < block_size * 2:
                time.sleep(0.005)
                continue

            samples = np.frombuffer(raw_bytes, dtype=np.int16)
            buffer.extend(samples)
            sample_counter += len(samples)

            if sample_counter >= step_size and len(buffer) >= window_size:
                sample_counter = 0
                batch_counter += 1
                file_log_batch_counter += 1
                window = np.array(buffer)[-window_size:]

                # Prepare WAV save
                wav_path = wav_dir / f"window_{batch_counter:03d}.wav"
                audio_bytes = window.astype(np.int16).tobytes()
                save_wave_file(str(wav_path), audio_bytes, sample_rate, sample_width=2,)
                logging.info(f"Saved batch wav {batch_counter}/{batch_size}")
                ts = datetime.now().isoformat(timespec='seconds')
                ts_array.append(ts)
                if batch_counter >= batch_size:
                    batch_predict(
                        wav_dir, current_log, scaler, ocsvm, svm,
                        components, model_threshold, sample_rate,
                        n_mfcc, max_frames, hop_length, tester_name, ts_array
                    )
                    batch_counter = 0

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down.")
    finally:
        if ser.is_open:
            ser.close()
            logging.info("Serial port closed.")

if __name__ == "__main__":
    main()
