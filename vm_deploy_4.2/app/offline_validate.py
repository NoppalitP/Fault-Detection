import argparse
import csv
import re
from pathlib import Path
import numpy as np
import yaml
import joblib
import librosa

from app.model import extract_features
from app.audio import compute_db, compute_top_frequencies


def load_cfg(base: Path):
    return yaml.safe_load(open(base/"config"/"config.yaml"))


def _parse_timestamp_from_filename(filename: str) -> str:
    """
    Extract timestamp from filenames like:
    - component_YYYYMMDD_HHMMSS.wav
    - elevator_audio_YYYYMMDD_HHMMSS.wav
    Returns ISO-like "YYYY-MM-DDTHH:MM:SS" or empty string if not found.
    """
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if not m:
        return ""
    ymd, hms = m.group(1), m.group(2)
    return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}T{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"


def predict_files(input_dir: Path, base: Path):
    cfg = load_cfg(base)
    ocsvm = joblib.load(base / cfg['models']['ocsvm'])
    log_reg = joblib.load(base / cfg['models']['log_reg'])
    # Optional scaler
    scaler = None
    try:
        scaler_path = cfg.get('models', {}).get('scaler')
        if scaler_path:
            scaler = joblib.load(base / scaler_path)
    except Exception:
        scaler = None

    sr = cfg['audio']['sample_rate']
    n_mfcc = cfg['mfcc']['n_mfcc']
    components = cfg['components']

    db_normal_max = cfg['db']['normal_max']
    db_anomaly_min = cfg['db']['anomaly_min']
    ocsvm_threshold = cfg['ocsvm']['threshold']
    calib_offset = cfg['db']['calib_offset']

    files = sorted([p for p in Path(input_dir).glob("*.wav")])
    # Original columns (removed OCSVMScore since it's the same as status_proba)
    base_columns = ["File","Timestamp","Component","Component_proba","Status","Status_proba","dB","TopFreq1","TopFreq2","TopFreq3","TopFreq4","TopFreq5"]
    # New columns
    # MFCC feature columns (13 columns)
    mfcc_columns = [f"MFCC{i+1}" for i in range(13)]
    # Combine all columns
    all_columns = base_columns  + mfcc_columns
    rows = [all_columns]

    for p in files:
        y, _ = librosa.load(str(p), sr=sr)
        feat = extract_features(y, sr, n_mfcc)
        X = np.asarray([feat], dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X)

        # classifier
        cls = int(log_reg.predict(X)[0])
        label = components[cls]
        # probability for the predicted class (fallback to max if classes not aligned)
        try:
            proba_row = np.asarray(log_reg.predict_proba(X)[0], dtype=float)
            if hasattr(log_reg, 'classes_'):
                classes = np.asarray(getattr(log_reg, 'classes_'))
                # classes_ could be ints [0..n-1] or other labels
                match_idx = np.where(classes == cls)[0]
                if match_idx.size:
                    prob = float(proba_row[int(match_idx[0])])
                else:
                    prob = float(np.max(proba_row))
            else:
                prob = float(np.max(proba_row))
        except Exception:
            prob = float('nan')

        # db and frequencies
        db = compute_db(y,
                method='ln',
                calib_offset=cfg['db']['calib_offset'],
                clamp_min=0)
        freqs = compute_top_frequencies(y, sr)

        # gating with ocsvm
        finite = np.isfinite(db)
        if finite and db < db_normal_max:
            status = "Normal"
            status_proba = ""
        elif finite and db > db_anomaly_min:
            status = "Anomaly"
            status_proba = ""
        else:
            score_val = float(ocsvm.decision_function(X)[0])
            status = "Normal" if score_val >= ocsvm_threshold else "Anomaly"
            status_proba = f"{score_val:.2f}"

        # Calculate component probability
        component_proba = f"{prob:.2f}" if np.isfinite(prob) else ""
        
        # Get frequencies (ensure we have at least 5 frequencies)
        freq4 = f"{freqs[3]:.1f}" if len(freqs) > 3 else ""
        freq5 = f"{freqs[4]:.1f}" if len(freqs) > 4 else ""
        
        # Get MFCC features (all 13 coefficients)
        mfcc_features = [f"{feat:.2f}" for feat in feat[:13]]  # Take first 13 MFCC coefficients

        # Original row data
        base_row = [
            p.name,
            _parse_timestamp_from_filename(p.name),
            label,
            component_proba,
            status,
            status_proba,
            f"{db:.1f}" if np.isfinite(db) else "",
            *[f"{f:.1f}" for f in (freqs if len(freqs) else [np.nan, np.nan, np.nan])][:5],
        ]
        # New columns data
        # MFCC features data
        mfcc_data = mfcc_features
        
        # Combine all data
        row = base_row  + mfcc_data
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing .wav files")
    parser.add_argument("--out", type=str, default="offline_results.csv")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    rows = predict_files(Path(args.input_dir), base)
    out_path = base / args.out
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


