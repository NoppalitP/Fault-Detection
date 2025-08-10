import numpy as np
import librosa
import wave
import sys


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
