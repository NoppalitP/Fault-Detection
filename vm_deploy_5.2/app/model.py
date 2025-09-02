import joblib, librosa
from pathlib import Path
from typing import Tuple, List
from app.audio import save_wave_file
import numpy as np
from app.audio import compute_db, compute_top_frequencies
import csv, logging , os , tempfile , time

# =========================
# Utilities
# =========================

def load_models(base: Path, cfg: dict):
    """
    Load only classifier and optional scaler (no OCSVM).
    Expect cfg['models']['classification_model'] and optional cfg['models']['scaler'].
    """
    lgb = joblib.load(base / cfg['models']['classification_model'])
    scaler = None
    try:
        scaler_path = cfg.get('models', {}).get('scaler')
        if scaler_path:
            scaler = joblib.load(base / scaler_path)
    except Exception:
        logging.exception("Failed to load scaler; continuing without it")
    return lgb, scaler


import numpy as np
import librosa

def extract_features(segment: np.ndarray, sr: int, n_mfcc: int = 13):
    """
    Extract MFCC-based features from a raw audio segment.

    Args:
        segment (np.ndarray): Audio samples
        sr (int): Sampling rate
        n_mfcc (int): Number of MFCC coefficients

    Returns:
        np.ndarray: Feature vector (mean + std of MFCCs)
    """
    # 1. MFCCs
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Combine into one vector
    features = np.hstack([
        mfccs_mean,
        mfccs_std
    ])

    return features



def preprocess_file(wav_path: Path, sample_rate: int, n_mfcc: int) -> Tuple:
    sig, _ = librosa.load(str(wav_path), sr=sample_rate)
    feat_vec = extract_features(sig, sample_rate, n_mfcc)
    return feat_vec, sig


def safe_write(rows, log_path, retries=5, delay=0.5):
    tmp_path = log_path.with_suffix(".tmp")
    
    # เขียนลง tmp ก่อน
    with open(tmp_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    # พยายาม replace ด้วย retry
    for attempt in range(retries):
        try:
            os.replace(tmp_path, log_path)
            return
        except PermissionError:
            time.sleep(delay)

    raise PermissionError(f"Could not replace {log_path} after {retries} retries")


# =========================
# Main batch predict
# =========================

def batch_predict(
    wav_dir: Path,
    log_path: Path,
    classifier,                     # e.g., LogisticRegression/LightGBM with predict_proba
    components: List[str],
    sample_rate: int,
    n_mfcc: int,
    tester_name: str,
    ts_array: List[str],
    threshold: float,
    calib_offset: float,
    ref_rms: float,
    scaler=None,
    cls_conf_min: float = 0.55      # ความมั่นใจขั้นต่ำสำหรับการตัดสินใจในแถบกลาง
):
    """
    Batch predict without OCSVM:
    - ใช้ dB thresholds ตัดสิน normal/anomaly
    - ถ้าอยู่ช่วงกลาง ใช้ความมั่นใจของ classifier
    """
    threshold  = float(threshold)

    files = sorted(wav_dir.glob("window_*.wav"))
    if not files:
        logging.warning("No files matched window_*.wav in %s", wav_dir)
        return

    n = min(len(files), len(ts_array))
    if n < len(files):
        logging.warning("ts_array has %d entries for %d files; truncating to %d.",
                        len(ts_array), len(files), n)
        files = files[:n]

    feats, sigs = [], []
    for f in files:
        try:
            feat, sig = preprocess_file(f, sample_rate, n_mfcc)
            feats.append(feat)
            sigs.append(sig)
        except Exception:
            logging.exception("Error preprocessing %s", f)
            feats.append(None)
            sigs.append(None)

    valid_idx = [i for i, v in enumerate(feats) if v is not None]
    if not valid_idx:
        logging.error("All feature extractions failed.")
        return

    X = np.asarray([feats[i] for i in valid_idx], dtype=np.float32)
    if scaler is not None:
        X = scaler.transform(X)

    # Predict classes + probabilities
    cls_pred = classifier.predict(X).astype(int)  # aligned to valid_idx
    try:
        probs_all = classifier.predict_proba(X)   # shape (k, n_classes)
    except Exception:
        probs_all = None

    # Compute dB & top5 freqs
    dbs, freqs_all = [], []
    for i in valid_idx:
        try:
            dbs.append(compute_db(sigs[i], calib_offset=calib_offset, ref_rms=ref_rms))
        except Exception:
            logging.exception("compute_db failed for %s", files[i])
            dbs.append(np.nan)
        try:
            freqs_all.append(compute_top_frequencies(sigs[i], sample_rate))
        except Exception:
            logging.exception("compute_top_frequencies failed for %s", files[i])
            freqs_all.append([])

    db_arr = np.asarray(dbs, dtype=float)
    finite = np.isfinite(db_arr)

    # NOTE: แก้บั๊กเดิมให้ถูกต้อง
    lt_mask = finite & (db_arr < threshold)    # ต่ำกว่าเกณฑ์ปกติ → Normal
    gt_mask = finite & (db_arr > threshold)   # สูงกว่าเกณฑ์ผิดปกติ → Anomaly
    mid_mask = finite & ~(lt_mask | gt_mask)       # ช่วงกลาง → ใช้ความมั่นใจ classifier

    f1 = "{:.2f}".format
    rows = []

    for k, i in enumerate(valid_idx):
        # เลือก label และ proba ของ label นั้น
        label_idx = int(cls_pred[k]) if np.ndim(cls_pred) == 1 else int(np.argmax(cls_pred[k]))
        label = components[label_idx] if 0 <= label_idx < len(components) else "unknown"

        # ความมั่นใจของคลาสที่ทำนาย
        label_proba = ""
        max_proba = None
        if probs_all is not None and probs_all.shape[0] > k:
            max_proba = float(np.max(probs_all[k]))
            if 0 <= label_idx < probs_all.shape[1]:
                label_proba = f1(probs_all[k, label_idx])

        # ตัดสินสถานะด้วย dB ก่อน แล้วค่อย fallback ที่ช่วงกลาง
        if lt_mask[k]:
            status = "Normal"
        elif gt_mask[k]:
            status = "Anomaly"
        else:
            # กลาง → ใช้ความมั่นใจของโมเดล
            if (max_proba is not None) and (max_proba >= cls_conf_min):
                # ยอมรับผล classifier
                # จะเลือก status = "Anomaly" เฉพาะบาง label ก็ได้ แต่ที่นี่กำหนด Normal โดย default
                # ปรับตามบริบทจริงได้ เช่น ถ้า label เป็น "environment" แล้ว dB ยังสูง อาจตั้ง "Review"
                status = "Normal"
            else:
                status = "Review"  # ไม่มั่นใจพอ

        # Top 5 freqs
        freqs = list(freqs_all[k][:5]) if len(freqs_all[k]) else []
        freqs += [0.0] * (5 - len(freqs))

        # dB (env adjustment)
        db_val = db_arr[k]
        if label == "environment":
            db_val -= 5.0
        db_val_str = f1(db_val)

        base_row = [
            ts_array[i],
            label,
            label_proba,
            status,
            "",                 # เดิมคือ status_proba (OCSVM) → เว้นว่าง
            db_val_str,
            *[f1(f) for f in freqs[:5]],
            tester_name,
        ]
        rows.append(base_row)

    # เขียนไฟล์แบบ atomic replace
    safe_write(rows, log_path)


    # Logging (คำนวณ label ใหม่ในลูป—แก้บั๊กเดิม)
    for k, i in enumerate(valid_idx):
        label_idx = int(cls_pred[k]) if np.ndim(cls_pred) == 1 else int(np.argmax(cls_pred[k]))
        label = components[label_idx] if 0 <= label_idx < len(components) else "unknown"

        if lt_mask[k]:
            status = "Normal"
        elif gt_mask[k]:
            status = "Anomaly"
        else:
            status = "Normal"
            if probs_all is not None and probs_all.shape[0] > k:
                if float(np.max(probs_all[k])) < cls_conf_min:
                    status = "Review"

        prob_str = ""
        if probs_all is not None and probs_all.shape[0] > k and 0 <= label_idx < probs_all.shape[1]:
            prob_str = f1(float(probs_all[k, label_idx]))

        db_val = db_arr[k]
        if label == "environment":
            db_val -= 5.0
        db_val_str = f1(db_val)

        logging.info(
            "%s: %s prob=%s %s dB=%s",
            files[i].name,
            label,
            prob_str,
            status,
            db_val_str,
        )

    # cleanup wav windows
    for wf in wav_dir.glob("window_*.wav"):
        try:
            wf.unlink()
        except Exception:
            logging.exception("Failed to delete %s", wf)