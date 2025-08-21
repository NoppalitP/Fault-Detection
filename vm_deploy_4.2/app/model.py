import joblib, librosa
from pathlib import Path
from typing import Tuple, List
from app.audio import save_wave_file
import numpy as np
from app.audio import compute_db, compute_top_frequencies
import csv, logging

# =========================
# Utilities
# =========================

def load_models(base: Path, cfg: dict):
    """Load OCSVM, classifier, and optional scaler (if configured)."""
    ocsvm = joblib.load(base / cfg['models']['ocsvm'])
    log_reg = joblib.load(base / cfg['models']['log_reg'])
    scaler = None
    try:
        scaler_path = cfg.get('models', {}).get('scaler')
        if scaler_path:
            scaler = joblib.load(base / scaler_path)
    except Exception:
        logging.exception("Failed to load scaler; continuing without it")
    return ocsvm, log_reg, scaler


def extract_features(segment, sr, n_mfcc):
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    feat_vec = mfccs.mean(axis=1)
    return feat_vec


def preprocess_file(wav_path: Path, sample_rate: int, n_mfcc: int) -> Tuple:
    sig, _ = librosa.load(str(wav_path), sr=sample_rate)
    feat_vec = extract_features(sig, sample_rate, n_mfcc)
    return feat_vec, sig


# =========================
# Main batch predict
# =========================

def batch_predict(
    wav_dir: Path, log_path: Path, ocsvm, log_reg, components: List[str],
    sample_rate: int, n_mfcc: int, tester_name: str, ts_array: List[str],
    db_normal_max: float, db_anomaly_min: float, ocsvm_threshold: float,
    calib_offset: float, method: str, ref_rms: float,
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

    # valid indices relative to the original files list
    valid_idx = [i for i, v in enumerate(feats) if v is not None]
    if not valid_idx:
        logging.error("All feature extractions failed.")
        return

    # Feature matrix X aligned to valid_idx order (k indexes this)
    X = np.asarray([feats[i] for i in valid_idx], dtype=np.float32)
    if scaler is not None:
        X = scaler.transform(X)

    # Classify all valid rows
    cls_out = log_reg.predict(X).astype(int)  # aligned to valid_idx (use k!)
    try:
        probs_all = log_reg.predict_proba(X)  # shape (len(valid_idx), n_classes)
    except Exception:
        probs_all = None

    # Compute dB and top frequencies per valid row (k index)
    dbs, freqs_all = [], []
    for i in valid_idx:
        try:
            dbs.append(compute_db(sigs[i], method=method, calib_offset=calib_offset, ref_rms=ref_rms))
        except Exception:
            logging.exception("compute_db failed for %s", files[i])
            dbs.append(np.nan)
        try:
            freqs_all.append(compute_top_frequencies(sigs[i], sample_rate))
        except Exception:
            logging.exception("compute_top_frequencies failed for %s", files[i])
            freqs_all.append([])

    # Threshold masks (arrays aligned to valid_idx → index with k)
    db_arr  = np.asarray(dbs, dtype=float)
    finite  = np.isfinite(db_arr)
    lt_mask = finite & (db_arr < DB_ANOMALY_MIN)   # below lower bound → Normal
    gt_mask = finite & (db_arr > DB_ANOMALY_MIN)  # above upper bound → Anomaly
    need_oc = finite & ~(lt_mask | gt_mask)       # only middle band needs OCSVM

    # Run OCSVM only where needed
    ocsvm_out   = np.zeros(len(valid_idx), dtype=int)
    ocsvm_score = np.full(len(valid_idx), np.nan, float)
    if np.any(lt_mask):
        idx_need = np.where(lt_mask)[0]                 # positions k
        oc_scores = ocsvm.decision_function(X[idx_need])
        ocsvm_score[idx_need] = oc_scores
        ocsvm_out[idx_need]   = np.where(oc_scores >= ocsvm_threshold, 1, -1)

    f1 = "{:.2f}".format
    rows = []

    for k, i in enumerate(valid_idx):
        # Decide Normal/Anomaly
        if lt_mask[k]:
            isnormal_str = "Normal"
        elif gt_mask[k]:
            isnormal_str = "Anomaly"
        else:
            isnormal_str = "Normal" if ocsvm_out[k] == 1 else "Anomaly"

        # Use k for arrays aligned to valid_idx (cls_out, probs_all)
        label_idx = int(cls_out[k]) if np.ndim(cls_out) == 1 else int(np.argmax(cls_out[k]))
        label = components[label_idx] if 0 <= label_idx < len(components) else "unknown"

        component_proba = ""
        if probs_all is not None and probs_all.shape[0] > k and 0 <= label_idx < probs_all.shape[1]:
            component_proba = f1(probs_all[k, label_idx])

        status_proba = f1(ocsvm_score[k]) if np.isfinite(ocsvm_score[k]) else ""

        # Frequencies (k)
        freqs = list(freqs_all[k][:5]) if len(freqs_all[k]) else []
        freqs += [0.0] * (5 - len(freqs))

        # dB (with environment adjustment)
        db_val = db_arr[k]
        if label == "environment":
            db_val -= 5.0
        db_val_str = f1(db_val)

        # ts_array & files are in original order → index with i
        base_row = [
            ts_array[i],
            label,
            component_proba,
            isnormal_str,
            status_proba,
            db_val_str,
            *[f1(f) for f in freqs[:5]],
            tester_name,
        ]
        rows.append(base_row)

    # Write once
    with open(log_path, "a", newline="") as fh:
        csv.writer(fh).writerows(rows)

    # Logging per row (mirror indices carefully)
    for k, i in enumerate(valid_idx):
        if lt_mask[k]:
            isnormal_str = "Normal"
        elif gt_mask[k]:
            isnormal_str = "Anomaly"
        else:
            isnormal_str = "Normal" if ocsvm_out[k] == 1 else "Anomaly"

        if probs_all is not None and probs_all.shape[0] > k and 0 <= int(cls_out[k]) < probs_all.shape[1]:
            prob_str = f1(float(probs_all[k, int(cls_out[k])]))
        else:
            prob_str = ""
        db_val = db_arr[k]
        if label == "environment":
            db_val -= 5.0
        db_val_str = f1(db_val)

        logging.info(
            "%s: %s prob=%s %s dB=%s prob_ocsvm=%s",
            files[i].name,
            components[int(cls_out[k])] if 0 <= int(cls_out[k]) < len(components) else "unknown",
            prob_str,
            isnormal_str,
            db_val_str,
            f1(ocsvm_score[k]) if np.isfinite(ocsvm_score[k]) else "",
        )

    # cleanup wav windows
    for wf in wav_dir.glob("window_*.wav"):
        try:
            wf.unlink()
        except Exception:
            logging.exception("Failed to delete %s", wf)
