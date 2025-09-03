# -*- coding: utf-8 -*-
"""
Windows XP-safe serial opener & diagnostics (NO pyserial)
- Helps fix: "[INFO] No serial port; retrying in 3s..."
- Uses Win32 API via ctypes. Logs exact CreateFile/GetLastError codes.
- Can probe available COM ports via `mode` command.

Usage examples:
  # 1) Just check which ports exist (via `mode` parsing)
  python xp_serial_port_open_diagnostics.py --probe

  # 2) Try to open COM5 at safe baud rates with verbose errors
  python xp_serial_port_open_diagnostics.py --ports COM5 --baud 921600 --try_baud_fallbacks 1 --once

  # 3) Run your normal loop (reads, packetizes) after it successfully opens
  python xp_serial_port_open_diagnostics.py --ports COM5,COM3 --baud 500000 --samples 512 \
      --out_dir C:\\OmegaAMC\\M5Stick\\drop --prefix saturn_db --rotate_secs 60

If you see errors like err=5 (ACCESS_DENIED), close apps using the port.
If err=2 (FILE_NOT_FOUND), COMx likely doesn't exist on this machine/session.
If err=87 (INVALID_PARAMETER), the driver rejected the baud; enable --try_baud_fallbacks.
"""
from __future__ import print_function

import os
import sys
import time
import math
import argparse
from datetime import datetime
import ctypes
from ctypes import wintypes

# -------------------- Win32 bindings --------------------
GENERIC_READ  = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_SHARE_READ  = 0x00000001
FILE_SHARE_WRITE = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

_k32 = ctypes.windll.kernel32

_k32.CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE
]
_k32.CreateFileW.restype = wintypes.HANDLE

_k32.ReadFile.argtypes = [
    wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID
]
_k32.ReadFile.restype = wintypes.BOOL

_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.CloseHandle.restype = wintypes.BOOL

class DCB(ctypes.Structure):
    _fields_ = [
        ("DCBlength", wintypes.DWORD),
        ("BaudRate", wintypes.DWORD),
        ("fBinary", wintypes.DWORD, 1),
        ("fParity", wintypes.DWORD, 1),
        ("fOutxCtsFlow", wintypes.DWORD, 1),
        ("fOutxDsrFlow", wintypes.DWORD, 1),
        ("fDtrControl", wintypes.DWORD, 2),
        ("fDsrSensitivity", wintypes.DWORD, 1),
        ("fTXContinueOnXoff", wintypes.DWORD, 1),
        ("fOutX", wintypes.DWORD, 1),
        ("fInX", wintypes.DWORD, 1),
        ("fErrorChar", wintypes.DWORD, 1),
        ("fNull", wintypes.DWORD, 1),
        ("fRtsControl", wintypes.DWORD, 2),
        ("fAbortOnError", wintypes.DWORD, 1),
        ("fDummy2", wintypes.DWORD, 17),
        ("wReserved", wintypes.WORD),
        ("XonLim", wintypes.WORD),
        ("XoffLim", wintypes.WORD),
        ("ByteSize", ctypes.c_ubyte),
        ("Parity", ctypes.c_ubyte),
        ("StopBits", ctypes.c_ubyte),
        ("XonChar", ctypes.c_char),
        ("XoffChar", ctypes.c_char),
        ("ErrorChar", ctypes.c_char),
        ("EofChar", ctypes.c_char),
        ("EvtChar", ctypes.c_char),
        ("wReserved1", wintypes.WORD),
    ]

class COMMTIMEOUTS(ctypes.Structure):
    _fields_ = [
        ("ReadIntervalTimeout", wintypes.DWORD),
        ("ReadTotalTimeoutMultiplier", wintypes.DWORD),
        ("ReadTotalTimeoutConstant", wintypes.DWORD),
        ("WriteTotalTimeoutMultiplier", wintypes.DWORD),
        ("WriteTotalTimeoutConstant", wintypes.DWORD),
    ]

BuildCommDCBW = _k32.BuildCommDCBW
BuildCommDCBW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(DCB)]
BuildCommDCBW.restype = wintypes.BOOL

SetCommState = _k32.SetCommState
SetCommState.argtypes = [wintypes.HANDLE, ctypes.POINTER(DCB)]
SetCommState.restype = wintypes.BOOL

SetCommTimeouts = _k32.SetCommTimeouts
SetCommTimeouts.argtypes = [wintypes.HANDLE, ctypes.POINTER(COMMTIMEOUTS)]
SetCommTimeouts.restype = wintypes.BOOL

SetupComm = _k32.SetupComm
SetupComm.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
SetupComm.restype = wintypes.BOOL

PurgeComm = _k32.PurgeComm
PurgeComm.argtypes = [wintypes.HANDLE, wintypes.DWORD]
PurgeComm.restype = wintypes.BOOL

GetLastError = _k32.GetLastError

PURGE_RXCLEAR = 0x0008

# -------------------- Helpers --------------------

def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def probe_ports_via_mode():
    """Parse `mode` output to list COM ports on XP."""
    try:
        out = os.popen("mode").read()
    except Exception:
        return []
    ports = []
    for line in out.splitlines():
        line = line.strip()
        # Expect lines like: "Status for device COM5:"
        if line.lower().startswith("status for device com") and line.endswith(":"):
            name = line.split(" ")[-1].rstrip(":")  # COM5
            ports.append(name.upper())
    return ports


class MinimalSerial(object):
    def __init__(self, port, baudrate=500000, timeout=0):
        self.port = port
        dev = u"\\\\.\\%s" % port
        h = _k32.CreateFileW(
            dev, GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None
        )
        if h == INVALID_HANDLE_VALUE:
            err = GetLastError()
            raise IOError("CreateFile(%s) failed, err=%d" % (port, err))
        self.handle = h

        SetupComm(self.handle, 1 << 15, 1 << 15)
        PurgeComm(self.handle, PURGE_RXCLEAR)

        dcb = DCB()
        dcb.DCBlength = ctypes.sizeof(DCB)
        cfg = u"baud=%d parity=N data=8 stop=1" % int(baudrate)
        if not BuildCommDCBW(cfg, ctypes.byref(dcb)):
            err = GetLastError()
            raise IOError("BuildCommDCBW failed, err=%d" % err)
        if not SetCommState(self.handle, ctypes.byref(dcb)):
            err = GetLastError()
            raise IOError("SetCommState failed, err=%d" % err)

        to = COMMTIMEOUTS()
        to.ReadIntervalTimeout = 1
        to.ReadTotalTimeoutMultiplier = 0
        to.ReadTotalTimeoutConstant = 0
        to.WriteTotalTimeoutMultiplier = 0
        to.WriteTotalTimeoutConstant = 0
        if not SetCommTimeouts(self.handle, ctypes.byref(to)):
            err = GetLastError()
            raise IOError("SetCommTimeouts failed, err=%d" % err)

    def read(self, n):
        buf = ctypes.create_string_buffer(n)
        nRead = wintypes.DWORD()
        ok = _k32.ReadFile(self.handle, buf, n, ctypes.byref(nRead), None)
        if not ok:
            err = GetLastError()
            raise IOError("ReadFile failed, err=%d" % err)
        return buf.raw[:nRead.value]

    def close(self):
        if getattr(self, 'handle', None):
            _k32.CloseHandle(self.handle)
            self.handle = None


def open_first_working_port(candidates, baud, try_baud_fallbacks=False):
    """Try ports; if err=87, test safer baud rates."""
    fallback_bauds = [115200, 230400, 460800, 500000, 921600]
    if baud not in fallback_bauds:
        fallback_bauds.insert(0, baud)

    for p in candidates:
        # first attempt with requested baud
        try:
            ser = MinimalSerial(port=p, baudrate=baud, timeout=0)
            print("[OK] Opened serial:", p, "@", baud)
            return ser
        except Exception as e:
            msg = str(e)
            print("[WARN] %s -> %s" % (p, msg))
            # error 87: try fallbacks if asked
            if try_baud_fallbacks and ("err=87" in msg or "INVALID_PARAMETER" in msg):
                for fb in fallback_bauds:
                    if fb == baud:
                        continue
                    try:
                        ser = MinimalSerial(port=p, baudrate=fb, timeout=0)
                        print("[OK] Opened serial:", p, "@", fb, "(fallback)")
                        return ser
                    except Exception as e2:
                        print("[WARN] %s @ %d -> %s" % (p, fb, e2))
            # try next port
            continue
    return None


# -------------- Optional: your packetizer & CSV writer (shortened) --------------
HEADER = b"\xAA\x55"

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
            idx = self._find(self.buf)
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
    def _find(b):
        for i in range(0, len(b) - 1):
            if b[i] == 0xAA and b[i+1] == 0x55:
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
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        tmp = os.path.join(self.out_dir, "%s-%s.csv.tmp" % (self.prefix, ts))
        return tmp, tmp[:-4]
    def _open_new(self):
        self.tmp_path, _ = self._new_paths()
        self.fh = open(self.tmp_path, "wb")
        header = b"timestamp,db\r\n"
        self.fh.write(header); self.fh.flush()
        self.start_time = time.time(); self.bytes_written = len(header)
    def write_row(self, ts, db):
        if self.fh is None: self._open_new()
        line = ("%s,%.3f\r\n" % (ts, db)).encode("ascii")
        self.fh.write(line); self.fh.flush(); self.bytes_written += len(line)
    def rotate(self):
        if self.fh is None: return
        try: self.fh.close()
        finally:
            _, final = self._new_paths()
            try: os.rename(self.tmp_path, final)
            except Exception as e: print("[WARN] Rename failed:", e)
        self.fh = None; self._open_new()

# dB helper (same as before)
import math

def int16_le_rms_dB(payload_bytes, calib_db):
    n = len(payload_bytes) // 2
    if n == 0: return -120.0
    ss = 0.0
    for i in range(0, len(payload_bytes), 2):
        lo = ord(payload_bytes[i:i+1]); hi = ord(payload_bytes[i+1:i+2])
        val = (hi << 8) | lo
        if val >= 0x8000: val -= 0x10000
        x = float(val) / 32768.0; ss += x * x
    scale = 1.002374467 / 0.33347884
    rms = math.sqrt(ss / float(n))
    rms_pa = max(rms * scale, 1e-12)
    db = 20.0 * math.log10(rms_pa / 20e-6)
    return db + calib_db

# -------------------- CLI / main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="List COM ports via `mode` and exit")
    ap.add_argument("--ports", default="COM5,COM3", help="Comma-separated list of COM ports to try")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--try_baud_fallbacks", type=int, default=1, help="Try common fallback bauds on err=87")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out_dir", default="Logs")
    ap.add_argument("--prefix", default="saturn_db")
    ap.add_argument("--rotate_secs", type=int, default=60)
    ap.add_argument("--once", action="store_true", help="Open port once and exit (diagnostic)")
    ap.add_argument("--calib_db", type=float, default=0.0)
    args = ap.parse_args()

    if args.probe:
        found = probe_ports_via_mode()
        print("[PROBE] mode reports:", ", ".join(found) if found else "(none)")
        return

    candidates = [p.strip().upper() for p in args.ports.split(',') if p.strip()]
    print("[INFO] Trying ports:", ", ".join(candidates), " @ baud=", args.baud)

    ser = open_first_working_port(candidates, args.baud, bool(args.try_baud_fallbacks))
    if ser is None:
        print("[ERR] Could not open any port. Tips: close Arduino/M5Burner/Putty; check `mode`; lower baud.")
        return

    if args.once:
        print("[OK] Opened", ser.port, "— diagnostic ok. Closing.")
        try: ser.close()
        except Exception: pass
        return

    writer = RotatingCsvWriter(args.out_dir, args.prefix, args.rotate_secs, 10 * 1024 * 1024)
    pkt = Packetizer(args.samples)

    db_accum = []
    last_write = time.time()
    try:
        print("[RUN] Reading from %s" % ser.port)
        while True:
            data = ser.read(args.chunk)
            frames = pkt.feed(data)
            for payload in frames:
                db = int16_le_rms_dB(payload, args.calib_db)
                db_accum.append(db)
            if time.time() - last_write >= 1.0 and db_accum:
                max_db = max(db_accum)
                print("dB",max_db)
                writer.write_row(now_iso(), max_db)
                db_accum = []
                last_write = time.time()
            if not data:
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n[EXIT] Stopping.")
    finally:
        try: ser.close()
        except Exception: pass
        try: writer.rotate()
        except Exception: pass

if __name__ == '__main__':
    main()
