import joblib , librosa
from pathlib import Path
from typing import Tuple, List
from app.audio import save_wave_file
import numpy as np
def load_models(base: Path, cfg: dict):
    iso  = joblib.load(base / cfg['models']['iso'])
    log_reg    = joblib.load(base / cfg['models']['log_reg'])
    
    return  iso, log_reg 

def extract_features(segment, sr, n_mfcc):
    #Spectral features
    #centroid  = librosa.feature.spectral_centroid(y=segment, sr=sr)
    # bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    # rolloff   = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    #flatness  = librosa.feature.spectral_flatness(y=segment)
    # contrast  = librosa.feature.spectral_contrast(y=segment, sr=sr)
    # zcr       = librosa.feature.zero_crossing_rate(y=segment)
    #rms       = librosa.feature.rms(y=segment)

    mfccs = librosa.feature.mfcc(y=segment, sr=22050, n_mfcc=n_mfcc)

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
    feat_vec = extract_features(sig,sample_rate,sig,n_mfcc)
    return feat_vec, sig



def batch_predict(wav_dir: Path, log_path: Path, scaler, iso, log_reg, components: List[str],
                  model_threshold: float, sample_rate: int, n_mfcc: int, max_frames: int,
                  hop_length: int, tester_name: str, ts_array: List[str]):
    from .audio import compute_db, compute_top_frequencies
    import csv, logging

    for idx, wav_file in enumerate(sorted(wav_dir.glob("window_*.wav"))):
        try:
            features, sig = preprocess_file(wav_file, scaler, sample_rate, n_mfcc, max_frames, hop_length)
            db = compute_db(sig)
            freqs = compute_top_frequencies(sig, sample_rate)
            if iso.predict(features)[0]:
                isnormal = "Normal"
            else:
                isnormal = "Anomaly"


            label = components[log_reg.predict(features)[0]]
            row = [ts_array[idx], label, isnormal, f"{db:.1f}", *[f"{f:.1f}" for f in freqs], tester_name]
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            logging.info(f"{wav_file.name}: {label} {isnormal} dB={db:.1f}")
        except Exception as e:
            logging.error(f"Error processing {wav_file}", exc_info=e)

    # cleanup
    for wf in wav_dir.glob("window_*.wav"):
        wf.unlink()
