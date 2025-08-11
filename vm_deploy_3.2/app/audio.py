import numpy as np
import librosa
import wave
import sys


def compute_db(sig: np.ndarray, calib_offset: float = 0.0) -> float:
    """
    Return approximate dB SPL of this window using mic sensitivity:
    SPM1423: -22 dBFS @ 94 dB SPL  => offset ~ +116 dB.
    Optionally add `calib_offset` from field calibration.
    """


    # Normalize PCM int16 to float32 in [-1, 1]
    if np.issubdtype(sig.dtype, np.integer):
        y = sig.astype(np.float32) / 32768.0
    else:
        y = sig.astype(np.float32, copy=False)

    # Frame RMS (librosa default frame_length=2048, hop_length=512)
    rms = librosa.feature.rms(y=y)[0]          # shape = (n_frames,)
    rms = np.clip(rms, 1e-12, None)            # avoid -inf

    # dBFS (RMS referenced to full-scale amplitude = 1.0)
    dbfs = librosa.amplitude_to_db(rms, ref=1.0)  # 20*log10(rms/1.0)

    # Convert to approx dB SPL using mic sensitivity + optional calibration
    dbspl = np.max(dbfs)  + float(calib_offset)  # use peak frame in the window
    return float(dbspl)



def compute_top_frequencies(sig: np.ndarray, sr: int, top_n: int = 3, min_freq: float = 400.0) -> np.ndarray:
    """
    คืนค่า array ของความถี่ (Hz) ที่มี magnitude สูงสุด จำนวน top_n
    แต่จะพิจารณาเฉพาะความถี่ >= min_freq เท่านั้น
    """
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1 / sr)
    mags = np.abs(S)

    # เลือกเฉพาะความถี่ที่ >= min_freq
    mask = f >= min_freq
    if not np.any(mask):
        return np.array([], dtype=float)

    masked_idx = np.where(mask)[0]
    masked_mags = mags[masked_idx]

    # ถ้าจำนวนที่ผ่านเงื่อนไข <= top_n ให้คืนทั้งหมด (เรียงจากมากไปน้อย)
    if len(masked_idx) <= top_n:
        order = np.argsort(masked_mags)[::-1]
        top_idx = masked_idx[order]
    else:
        # เลือก top_n แบบมีประสิทธิภาพ แล้วเรียงจากมากไปน้อย
        part = np.argpartition(masked_mags, -top_n)[-top_n:]
        order = part[np.argsort(masked_mags[part])[::-1]]
        top_idx = masked_idx[order]

    return f[top_idx]


def save_wave_file(filepath: str, audio_data: bytes, sample_rate: int, sample_width: int):
    try:
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
    except Exception as e:
        print(f"❌ Error saving file {filepath}: {e}", file=sys.stderr)
