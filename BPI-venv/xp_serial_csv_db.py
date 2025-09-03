# -*- coding: utf-8 -*-
"""
XP-safe Serial -> Rotating CSV (timestamp,db)
Python 3.4.3 (32-bit) + pyserial==3.0

Reads framed audio packets from serial:
  [0xAA, 0x55] + N * int16 (little-endian)
Computes RMS dBFS (with optional calibration offset) averaged over 2-second windows
and writes rows:
  timestamp_iso,db

Usage example:
  python xp_serial_csv_db.py --ports COM3,COM5 --baud 921600 --samples 512 \
      --out_dir C:\\OmegaAMC\\M5Stick\\drop --prefix saturn_db --rotate_secs 60 \
      --max_bytes 10485760 --calib_db -3.0

Dispatcher should watch only for *.csv (ignore *.csv.tmp).
"""
from __future__ import print_function

import os
import sys
import time
import math
import argparse
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
except Exception:
    print("pyserial is required (e.g., pip install pyserial==3.0)", file=sys.stderr)
    raise

HEADER = b"\xAA\x55"


def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def list_available_ports():
    ports = []
    try:
        for info in list(serial.tools.list_ports.comports()):
            if isinstance(info, tuple):
                ports.append(info[0])
            else:
                dev = getattr(info, "device", None) or getattr(info, "name", None)
                if dev:
                    ports.append(dev)
    except Exception:
        pass
    return ports


def open_first_working_port(candidates, baud):
    for p in candidates:
        try:
            ser = serial.Serial(port=p, baudrate=baud, timeout=0)
            print("[OK] Opened serial:", p)
            return ser
        except Exception:
            continue
    return None


class Packetizer(object):
    def __init__(self, samples_per_packet):
        self.samples_bytes = samples_per_packet * 2
        self.buf = bytearray()

    def feed(self, data):
        if not data:
            return []
        self.buf.extend(data)
        frames = []
        while True:
            idx = self._find_header(self.buf)
            if idx < 0:
                if len(self.buf) > 1:
                    self.buf[:] = self.buf[-1:]
                break
            if idx > 0:
                del self.buf[:idx]
            need = 2 + self.samples_bytes
            if len(self.buf) < need:
                break
            del self.buf[:2]
            payload = bytes(self.buf[:self.samples_bytes])
            del self.buf[:self.samples_bytes]
            frames.append(payload)
        return frames

    @staticmethod
    def _find_header(b):
        for i in range(0, len(b) - 1):
            if b[i] == 0xAA and b[i + 1] == 0x55:
                return i
        return -1


class RotatingCsvWriter(object):
    def __init__(self, out_dir, prefix, rotate_secs, max_bytes):
        self.out_dir = out_dir
        self.prefix = prefix
        self.rotate_secs = rotate_secs
        self.max_bytes = max_bytes
        self.fh = None
        self.tmp_path = None
        self.start_time = 0.0
        self.bytes_written = 0
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    def _new_paths(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp = os.path.join(self.out_dir, "%s-%s.csv.tmp" % (self.prefix, ts))
        final = tmp[:-4]
        return tmp, final

    def _open_new(self):
        self.tmp_path, _ = self._new_paths()
        self.fh = open(self.tmp_path, "wb")
        header = b"timestamp,db\r\n"
        self.fh.write(header)
        self.fh.flush()
        self.start_time = time.time()
        self.bytes_written = len(header)

    def _should_rotate(self):
        if self.rotate_secs > 0 and (time.time() - self.start_time) >= self.rotate_secs:
            return True
        if self.max_bytes > 0 and self.bytes_written >= self.max_bytes:
            return True
        return False

    def write_row(self, timestamp_iso, db_value):
        if self.fh is None:
            self._open_new()
        line = ("%s,%.3f\r\n" % (timestamp_iso, db_value)).encode("ascii")
        self.fh.write(line)
        self.fh.flush()
        self.bytes_written += len(line)
        if self._should_rotate():
            self.rotate()

    def rotate(self):
        if self.fh is None:
            return
        try:
            self.fh.close()
        finally:
            final = self.tmp_path[:-4]
            try:
                os.rename(self.tmp_path, final)
                print("[ROTATE] Finalized:", final)
            except Exception as e:
                print("[WARN] Rename failed:", e)
        self.fh = None
        self._open_new()


def int16_le_rms_dB(payload_bytes, calib_db):
    n = len(payload_bytes) // 2
    if n == 0:
        return -120.0
    ss = 0.0
    for i in range(0, len(payload_bytes), 2):
        lo = payload_bytes[i]
        hi = payload_bytes[i + 1]
        val = hi << 8 | lo
        if val >= 0x8000:
            val -= 0x10000
        x = float(val) / 32768.0
        ss += x * x
    scale = 1.002374467 / 0.33347884
    rms = math.sqrt(ss / float(n))
    rms_pa = max(rms * scale, 1e-12)
    db = 20.0 * math.log10(rms_pa / 20e-6)
    db_cal = db + calib_db
    print(db_cal)
    return db_cal



def main():
    ap = argparse.ArgumentParser(description="Serial -> CSV (timestamp,db)")
    ap.add_argument("--ports", default="COM3,COM5")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="saturn_db")
    ap.add_argument("--rotate_secs", type=int, default=60)
    ap.add_argument("--max_bytes", type=int, default=10 * 1024 * 1024)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--calib_db", type=float, default=0.0)
    args = ap.parse_args()

    if args.ports.strip().lower() == "auto":
        candidates = list_available_ports() or ["COM3", "COM5"]
    else:
        candidates = [p.strip() for p in args.ports.split(",") if p.strip()]

    writer = RotatingCsvWriter(args.out_dir, args.prefix, args.rotate_secs, args.max_bytes)
    pkt = Packetizer(args.samples)

    db_accum = []
    last_write = time.time()

    while True:
        ser = open_first_working_port(candidates, args.baud)
        if ser is None:
            print("[INFO] No serial port; retrying in 3s...")
            time.sleep(3)
            continue
        try:
            print("[RUN] Reading from %s @ %d" % (ser.port, args.baud))
            while True:
                data = ser.read(args.chunk)
                frames = pkt.feed(data)
                for payload in frames:
                    db = int16_le_rms_dB(payload, args.calib_db)
                    db_accum.append(db)
                if time.time() - last_write >= 1.0 and db_accum:
                    max_db = max(db_accum)
                    print(max_db)
                    writer.write_row(now_iso(), max_db)
                    db_accum = []
                    last_write = time.time()
                if not data:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print("\n[EXIT] Stopping.")
            try:
                ser.close()
            except Exception:
                pass
            try:
                writer.rotate()
            except Exception:
                pass
            break
        except Exception as e:
            print("[WARN] Serial loop error:", e)
            try:
                ser.close()
            except Exception:
                pass
            time.sleep(2)


if __name__ == "__main__":
    main()
