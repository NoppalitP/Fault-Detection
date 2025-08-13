import joblib , librosa
from pathlib import Path
from typing import Tuple, List
from app.audio import save_wave_file
import numpy as np
from app.audio import compute_db, compute_top_frequencies
import csv, logging
def load_models(base: Path, cfg: dict):
    """Load OCSVM, classifier, and optional scaler (if configured)."""
    ocsvm = joblib.load(base / cfg['models']['ocsvm'])
    log_reg = joblib.load(base / cfg['models']['log_reg'])
    try:
        scaler_path = cfg.get('models', {}).get('scaler')
        if scaler_path:
            scaler = joblib.load(base / scaler_path)
    except Exception:
        scaler = None
    return ocsvm, log_reg, scaler

def extract_features(segment, sr, n_mfcc):
    #Spectral features
    #centroid  = librosa.feature.spectral_centroid(y=segment, sr=sr)
    # bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    # rolloff   = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    #flatness  = librosa.feature.spectral_flatness(y=segment)
    # contrast  = librosa.feature.spectral_contrast(y=segment, sr=sr)
    # zcr       = librosa.feature.zero_crossing_rate(y=segment)
    #rms       = librosa.feature.rms(y=segment)

    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)

    # Combine all features (take mean across time axis)
    feat_vec = np.concatenate([
        #centroid.mean(axis=1),
        # bandwidth.mean(axis=1),
        # rolloff.mean(axis=1),
        #flatness.mean(axis=1),
        # contrast.mean(axis=1),
        # zcr.mean(axis=1),
        #rms.mean(axis=1),
        mfccs.mean(axis=1),
    ])

    return feat_vec

def preprocess_file(wav_path: Path,  sample_rate: int, n_mfcc: int) -> Tuple:
    sig, sr = librosa.load(str(wav_path), sr=sample_rate)
    feat_vec = extract_features(sig,sample_rate,n_mfcc)
    return feat_vec, sig



def batch_predict(
    wav_dir: Path, log_path: Path, ocsvm, log_reg, components: List[str],
    sample_rate: int, n_mfcc: int, tester_name: str, ts_array: List[str],
    db_normal_max: float, db_anomaly_min: float, ocsvm_threshold: float,
    calib_offset: float,
    scaler=None  # optional
):


    DB_NORMAL_MAX = float(db_normal_max)
    DB_ANOMALY_MIN = float(db_anomaly_min)

    files = sorted(wav_dir.glob("window_*.wav"))
    if not files:
        logging.warning("No files matched window_*.wav in %s", wav_dir)
        return

    n = min(len(files), len(ts_array))
    if n < len(files):
        logging.warning("ts_array has %d entries for %d files; truncating to %d.", len(ts_array), len(files), n)
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

    # เก็บเฉพาะที่ extract feature สำเร็จ
    valid_idx = [i for i, v in enumerate(feats) if v is not None]
    if not valid_idx:
        logging.error("All feature extractions failed.")
        return

    X = np.asarray([feats[i] for i in valid_idx], dtype=np.float32)
    if scaler is not None:
        X = scaler.transform(X)

    # ทำนายคลาสด้วย log_reg (ทุกรายการ)
    cls_out = log_reg.predict(X).astype(int)
    # ความน่าจะเป็นของคลาสที่ทำนาย (ถ้ามี predict_proba)
    probs_all = None
    try:
        probs_all = log_reg.predict_proba(X)
    except Exception:
        probs_all = None

    # คำนวณ dB / top-freq ต่อแถว (สำหรับทำกติกา override และบันทึก log)
    dbs = []
    freqs_all = []
    for i in valid_idx:
        dbs.append(compute_db(sigs[i],method='ln' ,calib_offset=-30,clamp_min=0))
        freqs_all.append(compute_top_frequencies(sigs[i], sample_rate))

    # --- เลือกเฉพาะแถวที่ "ต้องรัน" OCSVM ---
    db_arr   = np.asarray(dbs, dtype=float)
    finite   = np.isfinite(db_arr)
    lt_mask  = finite & (db_arr < DB_NORMAL_MAX)     # < 78  => Normal (ไม่รัน OCSVM)
    gt_mask  = finite & (db_arr > DB_ANOMALY_MIN)    # > 88  => Anomaly (ไม่รัน OCSVM)
    need_oc  = ~(lt_mask | gt_mask)                  # ช่วง 78–88 หรือ dB ไม่ finite

    ocsvm_out = np.zeros(len(valid_idx), dtype=int)  # เตรียมที่ไว้ (0 เป็นค่า placeholder)
    ocsvm_score = np.full(len(valid_idx), np.nan, dtype=float)
    if np.any(need_oc):
        ocsvm_scores = ocsvm.decision_function(X[need_oc])  # shape = (n_need,)
        ocsvm_score[need_oc] = ocsvm_scores
        ocsvm_out[need_oc] = np.where(ocsvm_scores >= ocsvm_threshold, 1, -1)
    # ------------------------------------------------

    f1 = "{:.2f}".format
    rows = []
    for k, i in enumerate(valid_idx):
        # ตัดสินสถานะตามกติกา dB ก่อนเสมอ
        if lt_mask[k]:
            isnormal_str = "Normal"
        elif gt_mask[k]:
            isnormal_str = "Anomaly"
        else:
            # ช่วงกำกวม => ใช้ผล OCSVM
            isnormal_str = "Normal" if ocsvm_out[k] == 1 else "Anomaly"

        label = components[cls_out[k]]
        
        # Calculate component probability
        component_proba = ""
        if probs_all is not None and 0 <= cls_out[k] < probs_all.shape[1]:
            component_proba = f1(probs_all[k, cls_out[k]])
        
        # Calculate status probability (OCSVM score converted to probability-like value)
        status_proba = ""
        if np.isfinite(ocsvm_score[k]):
            # Use raw OCSVM score without sigmoid conversion
            status_proba = f1(ocsvm_score[k])
        
        # Get frequencies (ensure we have at least 5 frequencies)
        freqs = freqs_all[k]

        
        # Get MFCC features (all 13 coefficients)
        mfcc_features = [f1(feat) for feat in feats[i][:13]]  # Take first 13 MFCC coefficients
        
        # Original row data
        base_row = [ts_array[i], label,component_proba, isnormal_str, status_proba, f1(db_arr[k]), *[f1(f) for f in freqs[:5]], tester_name]
        # New columns data
        #new_data = [component_proba, status_proba, freq4, freq5]
        # MFCC features data
        mfcc_data = mfcc_features
        
        # Combine all data
        row = base_row  + mfcc_data
        rows.append(row)

    # เขียนไฟล์ครั้งเดียว
    with open(log_path, "a", newline="") as fh:
        csv.writer(fh).writerows(rows)

    # logging ต่อแถว (ออปชัน)
    for k, i in enumerate(valid_idx):
        if lt_mask[k]:
            isnormal_str = "Normal"
        elif gt_mask[k]:
            isnormal_str = "Anomaly"
        else:
            isnormal_str = "Normal" if ocsvm_out[k] == 1 else "Anomaly"

        # probability for predicted class if available
        if probs_all is not None and 0 <= cls_out[k] < probs_all.shape[1]:
            prob_for_label = float(probs_all[k, cls_out[k]])
            prob_str = f1(prob_for_label)
        else:
            prob_str = ""

        logging.info(
            "%s: %s prob=%s %s dB=%s prob_ocsvm=%s",
            files[i].name,
            components[cls_out[k]],
            prob_str,
            isnormal_str,
            f1(db_arr[k]),
            f1(ocsvm_score[k]) if np.isfinite(ocsvm_score[k]) else "",
        )

    # cleanup
    for wf in wav_dir.glob("window_*.wav"):
        wf.unlink()

