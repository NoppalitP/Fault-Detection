import csv, time, logging, yaml ,gc
from datetime import datetime
from pathlib import Path
from collections import deque
import threading
import numpy as np

# app/main.py
from app.logger import setup_logging
from app.serial_handler import open_serial_with_retry
from app.utils import spinner_task
from app.model import load_models, batch_predict


def new_log_file(start_time: datetime, log_dir: Path, tester_name: str) -> Path:
    fname = start_time.strftime(f"{tester_name}_%Y%m%d_%H%M%S.csv")
    path = log_dir / fname
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(["Timestamp","Component","Status","dB","TopFreq1","TopFreq2","TopFreq3","Tester_id"])
    return path

def main():
    base = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load(open(base/"config"/"config.yaml"))
    # params from cfg
    sp_port  = cfg['serial']['port']
    baud     = cfg['serial']['baud_rate']
    sr        = cfg['audio']['sample_rate']
    blk_sz    = cfg['audio']['block_size']
    win_sz    = cfg['window']['size']
    step_sz   = cfg['window']['step']
    batch_sz  = cfg['batch']['size']
    tester    = cfg['testers']['name']
    comps     = cfg['components']
    n_mfcc    = cfg['mfcc']['n_mfcc']

    ocsvm, log_reg = load_models(base, cfg)

    # prepare dirs & logging
    log_dir = base / cfg['logging']['log_dir']; log_dir.mkdir(exist_ok=True, parents=True)
    wav_dir = base / cfg['batch']['wav_dir'];   wav_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(base/"app.log")
    logging.info("Starting monitoring")

    ser = open_serial_with_retry(sp_port, baud)
    if not ser:
        return

    buffer = deque(maxlen=win_sz)
    ts_arr = []
    batch_file_counter = 0  # ตัวนับไฟล์สำหรับ batch processing (รีเซ็ตทุก 30 ไฟล์)
    sample_counter = 0


    stop_event = threading.Event()
    spinner = threading.Thread(target=spinner_task, args=(stop_event,), daemon=True)
    spinner.start()

    try:
        while True:
  
            b1 = ser.read(1)
            if not b1 or b1[0] != 0xAA:
                continue
            b2 = ser.read(1)
            if not b2 or b2[0] != 0x55:
                continue

            raw_bytes = ser.read(blk_sz * 2)
            if len(raw_bytes) < blk_sz * 2:
                time.sleep(0.005)
                continue

            samples = np.frombuffer(raw_bytes, dtype=np.int16)
            buffer.extend(samples)
            sample_counter += len(samples)
            

            if sample_counter >= step_sz and len(buffer) >= win_sz:
                window = np.array(buffer)[-win_sz:]

                # Prepare WAV save
                wav_path = wav_dir / f"window_{batch_file_counter:03d}.wav"
                batch_file_counter += 1
                
                audio_bytes = window.astype(np.int16).tobytes()
                from .audio import save_wave_file
                save_wave_file(str(wav_path), audio_bytes, sr, sample_width=2)
                logging.info(f"Saving wave file {batch_file_counter} / {batch_sz}")
                ts_arr.append(datetime.now().isoformat(timespec='seconds'))
                
                # Run batch prediction ทุก 30 ไฟล์
                if batch_file_counter >= batch_sz:
                    curr_log = new_log_file(datetime.now(), log_dir, tester)
                    logging.info(f"Rotated log: {curr_log}")
                    batch_predict(
                        wav_dir, curr_log, ocsvm, log_reg, comps,
                        sr, n_mfcc, tester, ts_arr[-batch_sz:],
                        cfg['db']['normal_max'], cfg['db']['anomaly_min'],
                        cfg['ocsvm']['threshold'], cfg['db']['calib_offset']
                    )  # ส่งเฉพาะ timestamp 30 ตัวล่าสุด
                    batch_file_counter = 0  # รีเซ็ตตัวนับ batch
                    ts_arr = []
                    gc.collect()  # ล้างหน่วยความจำ
                    logging.info(f"Successfully logged: {curr_log}")
                    
                sample_counter = 0  # รีเซ็ต sample counter หลังประมวลผล window
                
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        if 0 < batch_file_counter < batch_sz:
            logging.info(f"Processing remaining {batch_file_counter} files before shutdown")
            batch_predict(
                wav_dir, curr_log, ocsvm, log_reg, comps,
                sr, n_mfcc, tester, ts_arr[-batch_sz:],
                cfg['db']['normal_max'], cfg['db']['anomaly_min'],
                cfg['ocsvm']['threshold'], cfg['db']['calib_offset']
            )
        stop_event.set()
        if ser.is_open:
            ser.close()
            logging.info("Serial closed")

if __name__=="__main__":
    main()