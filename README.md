## Fault-Detection

Audio-based fault and component detection for industrial equipment. The runtime app (`vm_deploy_3.2`) listens to streamed PCM audio from a microcontroller over serial, windows the signal, saves temporary WAVs, performs classification (component type) and anomaly gating (OCSVM + dB rules), and logs results to CSV.

### Key features

- **Real-time monitoring** from serial audio stream
- **Offline validation** against folders of WAV files
- **Hybrid decision**: logistic regression component classifier + OCSVM anomaly score, gated by configurable dB thresholds
- **Lightweight logs** with timestamp, label, status, dB, and top frequencies

---

## Repository structure

- `vm_deploy_3.2/`
  - `app/`
    - `main.py`: Real-time entry point (serial → window → WAV temp → batch predict → CSV log)
    - `offline_validate.py`: Offline CLI to evaluate a folder of WAVs into a CSV
    - `model.py`: Load models (joblib), feature extraction (MFCC), batch prediction with dB gating + OCSVM
    - `audio.py`: dB computation (with `calib_offset`), top-N frequency extraction (FFT), save WAV helper
    - `serial_handler.py`: Robust serial open with retry
    - `logger.py`: logging setup (file + stdout)
    - `utils.py`: spinner indicator
  - `config/config.yaml`: All runtime parameters (serial, audio, window, models, thresholds, batch, mfcc)
  - `model_files/`: `ocsvm_model.joblib`, `log_reg_model.joblib`
  - `Logs/`: CSV outputs rotated by batch
  - `myenv/`: Bundled Windows Python environment with dependencies (optional)
- `arduino_code/`: Microcontroller capture/streaming sketch (e.g., M5StickC Plus)
- `data/`: Sample datasets and recordings (optional)
- `requirements.txt` / `environment.yml`: Alternative dependency specs if not using `myenv`

---

## Quick start

### Option A: Use bundled Python (Windows)

1. Verify models exist in `vm_deploy_3.2/model_files/`:
   - `ocsvm_model.joblib`
   - `log_reg_model.joblib`
2. Set your serial port in `vm_deploy_3.2/config/config.yaml` at `serial.port` (e.g., `COM5`).
3. Real-time run:
   ```powershell
   .\vm_deploy_3.2\myenv\Scripts\python.exe vm_deploy_3.2\app\main.py
   ```
4. Offline validation:
   ```powershell
   .\vm_deploy_3.2\myenv\Scripts\python.exe vm_deploy_3.2\app\offline_validate.py data\component_data_train_test\test --out offline_results.csv
   ```

### Option B: Use your own environment

- With Conda:
  ```bash
  conda env create -f environment.yml
  conda activate fault-detection
  ```
- Or with pip:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```
  Run commands as in Option A (replace interpreter path accordingly).

---

## Configuration (`vm_deploy_3.2/config/config.yaml`)

- **serial**: `port` (e.g., `COM5`), `baud_rate`
- **audio**: `sample_rate`, `block_size` (bytes read per frame)
- **window**: `size`, `step` (in samples; typically `size = sr * duration`, `step = sr * hop`)
- **logging**: `log_dir` for CSV logs (auto-created)
- **components**: ordered list of labels for classifier outputs
- **models**: paths to `ocsvm` and `log_reg` joblib files (relative to `vm_deploy_3.2`)
- **db**: `calib_offset`, `normal_max`, `anomaly_min` (dB gating thresholds)
- **ocsvm**: `threshold` for decision_function
- **testers**: `name` used in log filenames
- **batch**: `size` (number of windows per batch), `wav_dir` (temporary WAV directory, cleared each batch)
- **mfcc**: `n_mfcc` feature dimension

Tip: Ensure `window.size` and `window.step` are consistent with `audio.sample_rate`.

---

## How it works

1. Stream handler reads frames from serial synchronized by header bytes `0xAA 0x55`.
2. Samples accumulate in a deque; when a window reaches `window.size` and `step` is met, it is saved as a temp WAV.
3. After `batch.size` windows are collected, `batch_predict`:
   - Loads audio, extracts MFCC features
   - Predicts component with logistic regression
   - Computes dB and top-3 frequencies
   - Applies dB gating: `< normal_max ⇒ Normal`, `> anomaly_min ⇒ Anomaly`, otherwise OCSVM decides via `threshold`
   - Appends rows to CSV and clears temp WAVs

---

## Usage

### Real-time

```powershell
./vm_deploy_3.2/myenv/Scripts/python.exe vm_deploy_3.2/app/main.py
```

Logs are written to `vm_deploy_3.2/Logs/TESTER_YYYYMMDD_HHMMSS.csv`.

### Offline

```powershell
./vm_deploy_3.2/myenv/Scripts/python.exe vm_deploy_3.2/app/offline_validate.py <wav_folder> --out offline_results.csv
```

Input folder should contain `.wav` files (mono implied by the runtime pipeline).

---

## Log format (CSV columns)

- `Timestamp`, `Component`, `Status`, `dB`, `TopFreq1`, `TopFreq2`, `TopFreq3`, `Tester_id`
- Status logic:
  - dB < `db.normal_max` ⇒ `Normal`
  - dB > `db.anomaly_min` ⇒ `Anomaly`
  - Otherwise ⇒ OCSVM `decision_function` vs `ocsvm.threshold`

---

## Testing plan (recommended)

- Config/I-O
  - Confirm `config.yaml` loads; `Logs/` and `batch.wav_dir` auto-create.
  - Set small `batch.size` (e.g., 3) to exercise rotation quickly.
- Unit
  - `audio.compute_db`: monotonic vs amplitude; respects `calib_offset`.
  - `audio.compute_top_frequencies`: 1 kHz sine detected; handles `< min_freq`.
  - `model.extract_features`: returns length `n_mfcc`.
  - `model.preprocess_file`: returns `(features, signal)` from a tiny WAV.
  - `model.batch_predict_fast`: with mock `log_reg`/`ocsvm`, checks CSV rows, dB gating, and temp WAV cleanup.
- Offline CLI
  - Run on sample folder; rows count equals files; Status follows gating rules.
- Integration (no hardware)
  - Monkeypatch `serial.Serial` to emit header and frames; run `main.py` with small batch; verify WAV rotation and logs.
- E2E (hardware)
  - Set `serial.port`; verify continuous logging, stable memory, no leftover temp files.

---

## Troubleshooting

- Cannot open serial: verify `serial.port` (e.g., `COMx`) and permissions; device connected; baud matches firmware.
- Missing models: check `vm_deploy_3.2/model_files/` paths in `config.yaml`.
- Librosa/NumPy/Sklearn mismatch: prefer the bundled `myenv` or match versions in `requirements.txt`/`environment.yml`.
- No WAVs processed in batch: ensure `window.size/step` align with `audio.sample_rate` and streaming actually delivers frames.

---

## License

Internal/educational project. Add your preferred license here if distributing.
