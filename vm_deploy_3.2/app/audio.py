import numpy as np
import wave
import sys

def calcDecibell(athiwat: float) -> float:
    """
    จำลองสูตรใน C++: 8.6859 * ln(athiwat) + 25.6699
    หมายเหตุ: ใช้ ln (natural log)
    """
    athiwat = max(float(athiwat), 1e-12)  # กัน log(0)
    return 8.6859 * np.log(athiwat) + 25.6699

def _rms_counts(
    sig: np.ndarray,
    *,
    gain_factor: float,
    subtract_dc: bool,
) -> float:
    """
    เตรียมสัญญาณให้อยู่ในหน่วย "counts" แบบ int16 แล้วคำนวณ RMS
    - ถ้า sig เป็น float [-1,1] จะคูณ 32768 ให้เป็นสเกล int16
    - (เลือกได้) ลบ DC ก่อน
    - คูณ gain_factor เพื่อเลียนแบบเฟิร์มแวร์
    """
    x = sig.astype(np.float32, copy=False)
    if np.issubdtype(sig.dtype, np.floating):
        x *= 32768.0
    if subtract_dc:
        x = x - np.mean(x)
    x *= float(gain_factor)
    rms = float(np.sqrt(np.mean(np.maximum(x * x, 0.0))))
    return max(rms, 1e-12)
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
    calib_offset: float = 0.0,
    *,
    method: str = "ln",           # "ref" หรือ "ln"
    gain_factor: float = 3.0,      # ตรงกับ GAIN_FACTOR ในตัวอย่าง C++
    ref_rms: float = 1000.0,       # ตรงกับ REF ในตัวอย่าง C++
    subtract_dc: bool = True,
    clamp_min: float  = 0.0, # เลือก clamp ขั้นต่ำ (เช่น 0 dB)
) -> float:
    """
    คำนวณ dB จากสัญญาณหนึ่งหน้าต่าง (window)
    - method="ref": dB = 20*log10(rms / ref_rms)
    - method="ln" : dB = calcDecibell(rms)
    จากนั้นบวก calib_offset และแปลงเชิงเส้นเป็นค่าที่รายงาน; สุดท้าย clamp ขั้นต่ำหากกำหนด
    """
    sig_ravel = np.ravel(sig)
    rms_counts = _rms_counts(sig_ravel, gain_factor=gain_factor, subtract_dc=subtract_dc)

    if method == "ref":
        db = 20.0 * np.log10(rms_counts / float(ref_rms)) +116
    elif method == "ln":
        db = calcDecibell(rms_counts)
    else:
        raise ValueError("method ต้องเป็น 'ref' หรือ 'ln'")

    # คาลิเบรตภาคสนามด้วยออฟเซ็ต
    db += float(calib_offset)
    # แปลงเชิงเส้นเป็นค่าที่รายงาน
    db = float(1.177 * db - 38.506)
    # Clamp ขั้นต่ำถ้าต้องการ (กับค่าหลังแปลง)
    if clamp_min is not None:
        db = max(db, float(clamp_min))
    return db




def compute_top_frequencies(sig: np.ndarray, sr: int, top_n: int = 3, min_freq: float = 400.0) -> np.ndarray:
    """
    คืนค่า array ของความถี่ (Hz) ที่มี magnitude สูงสุด จำนวน top_n
    แต่จะพิจารณาเฉพาะความถี่ >= min_freq เท่านั้น
    """
    sig = np.ravel(sig)
    if sig.size == 0:
        return np.array([], dtype=float)
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
