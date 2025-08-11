import argparse
import csv
import re
from pathlib import Path
import numpy as np
import yaml
import joblib
import librosa

from model import extract_features
from audio import compute_db, compute_top_frequencies


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

    sr = 22050
    n_mfcc = cfg['mfcc']['n_mfcc']
    components = cfg['components']

    db_normal_max = cfg['db']['normal_max']
    db_anomaly_min = cfg['db']['anomaly_min']
    ocsvm_threshold = cfg['ocsvm']['threshold']
    calib_offset = cfg['db']['calib_offset']

    files = sorted([p for p in Path(input_dir).glob("*.wav")])
    rows = [[
        "File","Timestamp","Component","Prob","Status","dB","OCSVMScore",
        "TopFreq1","TopFreq2","TopFreq3"
    ]]

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
        db = compute_db(y, calib_offset)
        freqs = compute_top_frequencies(y, sr)

        # gating with ocsvm
        finite = np.isfinite(db)
        if finite and db < db_normal_max:
            status = "Normal"
            score = ""
        elif finite and db > db_anomaly_min:
            status = "Anomaly"
            score = ""
        else:
            score_val = float(ocsvm.decision_function(X)[0])
            status = "Normal" if score_val >= ocsvm_threshold else "Anomaly"
            score = f"{score_val:.1f}"

        rows.append([
            p.name,
            _parse_timestamp_from_filename(p.name),
            label,
            f"{prob:.1f}" if np.isfinite(prob) else "",
            status, f"{db:.1f}" if np.isfinite(db) else "",
            score,
            *[f"{f:.1f}" for f in (freqs if len(freqs) else [np.nan, np.nan, np.nan])][:3],
        ])

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


