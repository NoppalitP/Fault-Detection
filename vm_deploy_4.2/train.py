import os, json, joblib
from pathlib import Path
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GroupShuffleSplit

# --------------------- CONFIG ---------------------
DATA_DIR = r"D:\NOPPALIT\Fault-Detection\all_tester"
PATTERNS = {
    0: "mast_*.wav",
    1: "elevator_*.wav",
    2: "gripper_*.wav",
    3: "shuttle_*.wav",
    4: "environment_*.wav",
    5: "mast_bearing_broken_*.wav"
}
SR = 22050
SEG_DUR = 2.0
OVERLAP = 1.0
N_MFCC = 13
BP_LO, BP_HI = 400.0, 4000.0
PREEMPH = 0.97
N_FFT = 2048
HOP = 512
RAND_SEED = 1337
VAL_RATIO = 0.2
USE_EMA_CMVN = False      # <<< BEST for deploy with EMA
EMA_ALPHA = 0.02      # 0.01–0.05 reasonable
OUT_DIR = Path("/content/models"); OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RAND_SEED)

# --------------------- IO & Segmentation ---------------------
from pathlib import Path

def load_signals(base_path: str, pattern: str, sr: int = SR):
    items = []
    for p in sorted(Path(base_path).glob(pattern)):
        try:
            y, _ = librosa.load(str(p), sr=sr)
            items.append((y.astype(np.float32), p.stem))
        except Exception as e:
            print(f"[WARN] load failed {p}: {e}")
    return items


def segment_signal(y: np.ndarray, sr: int = SR, seg_dur: float = SEG_DUR, overlap: float = OVERLAP):
    seg_len = int(round(seg_dur * sr))
    hop = max(1, seg_len - int(round(overlap * sr)))
    N = len(y)
    out = [y[i:i+seg_len] for i in range(0, max(0, N - seg_len + 1), hop)]
    if N >= seg_len and (N - seg_len) % hop != 0:
        out.append(y[-seg_len:])
    return out

# --------------------- DSP (Pre‑emph + Band‑pass) ---------------------
from scipy.signal import butter, sosfiltfilt

def pre_emphasis(x: np.ndarray, coeff: float = PREEMPH) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coeff * x[:-1]
    return y


def _bp_sos(sr: int, lo: float, hi: float, order: int = 4):
    nyq = 0.5 * sr
    lo_n = max(1.0, lo) / nyq
    hi_n = min(hi, nyq - 1.0) / nyq
    if not (0.0 < lo_n < hi_n < 1.0):
        raise ValueError(f"Invalid band ({lo}, {hi}) for sr={sr}")
    return butter(order, [lo_n, hi_n], btype="band", output="sos")


def bandpass(x: np.ndarray, sr: int, lo: float = BP_LO, hi: float = BP_HI, order: int = 4) -> np.ndarray:
    return sosfiltfilt(_bp_sos(sr, lo, hi, order), np.asarray(x, dtype=np.float32))

# --------------------- Frame‑wise EMA‑CMVN ---------------------

def ema_cmvn_frames(M: np.ndarray, alpha: float = EMA_ALPHA, eps: float = 1e-6) -> np.ndarray:
    """M: (C, T) MFCC frames. Online EMA mean/var per channel across time.
    Returns normalized (C, T)."""
    C, T = M.shape
    out = np.empty_like(M, dtype=np.float32)
    mu = np.zeros(C, dtype=np.float32)
    var = np.ones(C, dtype=np.float32)
    warmup = True
    for t in range(T):
        x = M[:, t].astype(np.float32)
        if warmup:
            mu = x
            var = np.ones(C, dtype=np.float32)
            warmup = False
        else:
            mu = (1 - alpha) * mu + alpha * x
            var = (1 - alpha) * var + alpha * (x - mu) ** 2
        out[:, t] = (x - mu) / np.sqrt(var + eps)
    return out

# --------------------- Feature extraction (Train = Deploy) ---------------------

def features_from_segment(seg: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    x = pre_emphasis(seg)
    x = bandpass(x, sr)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP)  # (C,T)

    if USE_EMA_CMVN:
        mfcc_n = ema_cmvn_frames(mfcc, alpha=EMA_ALPHA)
        # Warm-up: ตัด ~10% เฟรม แต่ไม่น้อยกว่า 3 และไม่เกิน 20
        T = mfcc_n.shape[1]
        W = min(20, max(3, int(0.10 * T)))
        mf = mfcc_n[:, W:] if T > W else mfcc_n

        # ชุดเบา/เสถียร (แนะนำ): mean + Q25 + Q75
        feat = np.concatenate([
            mf.mean(axis=1),
            np.percentile(mf, 25, axis=1),
            np.percentile(mf, 75, axis=1),
        ]).astype(np.float32)  # 3*C
    else:
        dmfcc = librosa.feature.delta(mfcc, width=5, order=1, mode="nearest")
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        logE = float(np.log(rms + 1e-12))
        feat = np.concatenate([
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            dmfcc.mean(axis=1),
            [logE],
        ]).astype(np.float32)  # 40D
    return feat


# --------------------- Build dataset ---------------------

def build_dataset(base_dir: str, patterns: dict):
    X_list, y_list, g_list = [], [], []
    for label, pattern in patterns.items():
        for sig, stem in load_signals(base_dir, pattern):
            for seg in segment_signal(sig, sr=SR, seg_dur=SEG_DUR, overlap=OVERLAP):
                X_list.append(features_from_segment(seg, SR, N_MFCC))
                y_list.append(int(label))
                g_list.append(stem)  # กลุ่มตามไฟล์
    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=int)
    groups = np.asarray(g_list)
    return X, y, groups


X, y, groups = build_dataset(DATA_DIR, PATTERNS)
print("Dataset:",X[1].shape , X.shape, y.shape, "#groups=", len(np.unique(groups)))

# --------------------- Group‑aware split ---------------------
RAND = RAND_SEED
from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=RAND_SEED)
train_idx, val_idx = next(splitter.split(X, y, groups))
X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
X_va, y_va, g_va = X[val_idx],   y[val_idx],   groups[val_idx]
print("Train:", X_tr.shape, " Val:", X_va.shape)

# --------------------- Normalization block ---------------------
scaler = None
if USE_EMA_CMVN:
    # Already normalized at frame level → DO NOT apply global scaler
    X_tr_n, X_va_n = X_tr, X_va
else:
    scaler = StandardScaler().fit(X_tr)
    X_tr_n = scaler.transform(X_tr)
    X_va_n = scaler.transform(X_va)

# --------------------- Classifier ---------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import OneClassSVM
import joblib, json
import numpy as np

# --------------------- Classifiers ---------------------
models = {
    "log_reg": LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1,class_weight="balanced", random_state=RAND_SEED),
    "rf": RandomForestClassifier(n_estimators=200, random_state=RAND_SEED, n_jobs=-1),
    "gb": GradientBoostingClassifier(random_state=RAND_SEED),
    "dt": DecisionTreeClassifier(random_state=RAND_SEED),
        "lgbm": LGBMClassifier(
        objective="multiclass",
        n_estimators=400,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_Pamples=5,
        min_split_gain=0.0,
        max_bin=255,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RAND,
        verbose=-1
    ),
}

# --------------------- Train & Eval ---------------------
for name, clf in models.items():
    clf.fit(X_tr_n, y_tr)

    y_pred = clf.predict(X_va_n)
    acc = accuracy_score(y_va, y_pred)
    f1m = f1_score(y_va, y_pred, average='macro')
    print(f"[{name}] Acc: {acc:.4f}  F1m: {f1m:.4f}")
    print(classification_report(y_va, y_pred, digits=4))
    print(confusion_matrix(y_va, y_pred))

    # save each model
    joblib.dump(clf, OUT_DIR / f"{name}.joblib")

# --------------------- OCSVM ---------------------
# (A) Global
ocsvm_global = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm_global.fit(X_tr_n)
joblib.dump(ocsvm_global, OUT_DIR/"ocsvm_global.joblib")

# (B) Per-class
per_cls = {}
for c in np.unique(y_tr):
    Xc = X_tr_n[y_tr == c]
    if len(Xc) >= 20:
        oc = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale').fit(Xc)
        per_cls[int(c)] = oc
if per_cls:
    joblib.dump(per_cls, OUT_DIR/"ocsvm_per_component.joblib")

# --------------------- Save Scaler & Config ---------------------
if scaler is not None:
    joblib.dump(scaler, OUT_DIR/"scaler.joblib")

with open(OUT_DIR/"norm_config.json", "w") as f:
    json.dump({
        "use_ema_cmvn": int(USE_EMA_CMVN),
        "ema_alpha": float(EMA_ALPHA),
        "sr": SR, "preemph": PREEMPH, "bp": [BP_LO, BP_HI],
        "n_fft": N_FFT, "hop": HOP, "n_mfcc": N_MFCC
    }, f)

print("All models saved to:", OUT_DIR)
