import joblib , librosa
from pathlib import Path
from typing import Tuple, List
from app.audio import reduce_noise, extract_mfcc, pad_mfcc

def load_models(base: Path, cfg: dict):
    scaler = joblib.load(base / cfg['models']['scaler'])
    ocsvm  = joblib.load(base / cfg['models']['ocsvm'])
    svm    = joblib.load(base / cfg['models']['log_reg'])
    expected_dim = scaler.mean_.shape[0]
    return scaler, ocsvm, svm, expected_dim

def preprocess_file(wav_path: Path, scaler, sample_rate: int, n_mfcc: int, max_frames: int, hop_length: int) -> Tuple:
    sig, sr = librosa.load(str(wav_path), sr=sample_rate)
    denoised = reduce_noise(sig, sr=sr)
    mf = extract_mfcc(denoised, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mf_fixed = pad_mfcc(mf, max_frames)
    flat = mf_fixed.flatten()[None, :]
    features = scaler.transform(flat)
    return features, denoised

def batch_predict(wav_dir: Path, log_path: Path, scaler, ocsvm, svm, components: List[str],
                  model_threshold: float, sample_rate: int, n_mfcc: int, max_frames: int,
                  hop_length: int, tester_name: str, ts_array: List[str]):
    from .audio import compute_db, compute_top_frequencies
    import csv, logging

    for idx, wav_file in enumerate(sorted(wav_dir.glob("window_*.wav"))):
        try:
            features, sig = preprocess_file(wav_file, scaler, sample_rate, n_mfcc, max_frames, hop_length)
            db = compute_db(sig)
            freqs = compute_top_frequencies(sig, sample_rate)
            score = ocsvm.decision_function(features)[0]
            status = "NORMAL" if score >= model_threshold else "ANOMALY"
            label = components[svm.predict(features)[0]]
            row = [ts_array[idx], label, status, f"{db:.1f}", *[f"{f:.1f}" for f in freqs], tester_name]
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            logging.info(f"{wav_file.name}: {label} {status} dB={db:.1f}")
        except Exception as e:
            logging.error(f"Error processing {wav_file}", exc_info=e)

    # cleanup
    for wf in wav_dir.glob("window_*.wav"):
        wf.unlink()
