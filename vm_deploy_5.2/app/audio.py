import numpy as np
import wave
import sys


def signal_rms(signal: np.ndarray, subtract_dc: bool = False) -> float:
    """
    คำนวณ RMS แบบสากล
    - signal: np.ndarray (float หรือ int ก็ได้)
    - subtract_dc: ถ้า True จะลบค่าเฉลี่ย (DC offset) ก่อน
    """
    x = signal.astype(np.float64, copy=False)  # ใช้ float64 เพื่อความแม่น
    if subtract_dc:
        x -= np.mean(x)
    return np.sqrt(np.mean(x**2))

def compute_db(
    sig: np.ndarray,
    ref_rms: float = 20e-6,       # ตรงกับ REF ในตัวอย่าง C++
    subtract_dc: bool = True,
    calib_offset: float = 0.0
) -> float:
    """
    คำนวณ dB จากสัญญาณหนึ่งหน้าต่าง (window)
    - method="ref": dB = 20*log10(rms / ref_rms)
    จากนั้นบวก calib_offset และแปลงเชิงเส้นเป็นค่าที่รายงาน; สุดท้าย clamp ขั้นต่ำหากกำหนด
    """
    sig_ravel = np.ravel(sig)
        # === Calibration scale factor ===
    scale = 1.002374467 / 0.33347884  # ค่านี้มาจากการวัด tone ที่รู้ SPL
    rms = float(signal_rms(sig_ravel, subtract_dc=subtract_dc))
    scale = float(scale)
    rms_pa = max(rms * scale, 1e-12)
    ref_rms = float(ref_rms)
    db = 20.0 * np.log10(rms_pa / ref_rms)

    # คาลิเบรตภาคสนามด้วยออฟเซ็ต
    # Clamp ขั้นต่ำถ้าต้องการ (กับค่าหลังแปลง)
    # if clamp_min is not None:
    #     db = max(db, float(clamp_min))
    return db + calib_offset


import numpy as np
from typing import Optional

def compute_top_frequencies(
    sig: np.ndarray,
    sr: int,
    top_n: int = 5,
    min_freq: float = 300.0,
    min_separation_hz: float = 50,
    nfft: int = 8192,   # <-- เพิ่มตัวเลือก NFFT
) -> np.ndarray:
    x = np.ravel(sig).astype(float)
    if x.size == 0:
        return np.array([], dtype=float)

    # เลือก NFFT: ถ้าไม่กำหนด ใช้ความยาวสัญญาณ
    N = x.size
    nfft = int(nfft or N)

    # DC remove + Hann
    x = x - x.mean()
    w = np.hanning(N)
    # ถ้า nfft > N จะเป็น zero-padding (ละเอียดขึ้นเฉพาะ grid ไม่เพิ่ม true resolution)
    X = np.fft.rfft(x * w, n=nfft)
    mags = np.abs(X)
    f = np.fft.rfftfreq(nfft, 1 / sr)

    # เลือกเฉพาะ f >= min_freq และกันขอบ (ต้องมี k-1,k+1)
    idx = np.where((f >= min_freq))[0]
    idx = idx[(idx > 0) & (idx < len(f) - 1)]
    if idx.size == 0:
        return np.array([], dtype=float)

    # default การกันพีกติดกัน: อย่างน้อย 1 bin
    if min_separation_hz is None:
        min_separation_hz = sr / nfft

    # candidates มากกว่าที่ต้องการ แล้วคัดด้วย NMS
    take = min(top_n * 8, idx.size)
    cand = idx[np.argpartition(mags[idx], -take)[-take:]]
    cand = cand[np.argsort(mags[cand])[::-1]]

    def interp_freq(k: int) -> float:
        m1, m0, p1 = mags[k-1], mags[k], mags[k+1]
        denom = (m1 - 2.0*m0 + p1)
        delta = 0.0 if denom == 0.0 else 0.5 * (m1 - p1) / denom
        return (k + delta) * sr / nfft

    pick_freqs = []
    for k in cand:
        f_hat = interp_freq(k)
        if all(abs(f_hat - pf) >= min_separation_hz for pf in pick_freqs):
            pick_freqs.append(f_hat)
        if len(pick_freqs) >= top_n:
            break

    return np.array(pick_freqs, dtype=float)



def save_wave_file(filepath: str, audio_data: bytes, sample_rate: int, sample_width: int):
    try:
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
    except Exception as e:
        print(f"❌ Error saving file {filepath}: {e}", file=sys.stderr)
