import numpy as np
import librosa
import noisereduce as nr
import wave
import sys

def reduce_noise(sig: np.ndarray, sr: int) -> np.ndarray:
    return nr.reduce_noise(y=sig, sr=sr)

def extract_mfcc(sig: np.ndarray, sr: int, n_mfcc: int = 40, hop_length: int = 512) -> np.ndarray:
    return librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

def pad_mfcc(mfcc: np.ndarray, max_frames: int) -> np.ndarray:
    n_mfcc, n_frames = mfcc.shape
    if n_frames < max_frames:
        return np.pad(mfcc, ((0, 0), (0, max_frames - n_frames)), mode='constant')
    return mfcc[:, :max_frames]

def compute_db(sig: np.ndarray) -> float:
    rms = librosa.feature.rms(y=sig)[0]
    dbfs = librosa.amplitude_to_db(rms, ref=1)
    mic_offset = 94 - (-22) + 54 - 82
    return float(np.max(dbfs + mic_offset))

def compute_top_frequencies(sig: np.ndarray, sr: int, top_n: int = 3) -> np.ndarray:
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1/sr)
    mags = np.abs(S)
    idx = np.argsort(mags)[-top_n:][::-1]
    return f[idx]

def save_wave_file(filepath: str, audio_data: bytes, sample_rate: int, sample_width: int):
    try:
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
    except Exception as e:
        print(f"❌ Error saving file {filepath}: {e}", file=sys.stderr)
