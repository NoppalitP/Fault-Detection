# -*- coding: utf-8 -*-
"""
Real-time frequency analysis, aligned with vm_deploy_3.2 runtime:
- Reads config from vm_deploy_3.2/config/config.yaml
- Opens serial via app.serial_handler.open_serial_with_retry
- Uses app.logger for consistent console output
- Uses app.utils spinner for connect/wait UX and stderr logs to avoid interleaving
"""
import numpy as np
from datetime import datetime
from collections import deque
import time
import sys
import threading
import queue
from pathlib import Path
import yaml
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from contextlib import nullcontext

# Resolve vm_deploy_3.2 package for shared utilities
BASE = Path(__file__).resolve().parent.parent / "vm_deploy_3.2"
sys.path.insert(0, str(BASE))

from app.logger import setup_logging
from app.serial_handler import open_serial_with_retry
from app.utils import spinner, start_spinner, stop_spinner

# === CONFIGURATION (from vm_deploy_3.2/config/config.yaml) ==================
CFG = yaml.safe_load(open(BASE / "config" / "config.yaml", "r"))

# --- Serial Port Settings ---
SERIAL_PORT = CFG['serial']['port']
BAUD_RATE = CFG['serial']['baud_rate']

# --- Audio Settings ---
SAMPLE_RATE = CFG['audio']['sample_rate']
CHANNELS = 1
SAMPLE_WIDTH = 2

# --- Recording Loop Settings ---
BLOCK_SIZE = CFG['audio']['block_size']

# --- Frequency Analysis Settings ---
WINDOW_DURATION = 1.0
UPDATE_INTERVAL = 50
FFT_SIZE = 2048
FREQ_RANGE = (0, SAMPLE_RATE / 2)

# UI options
UI_CFG = CFG.get('ui', {})
SPINNER_ENABLED = bool(UI_CFG.get('spinner_enabled', True))
SPINNER_INTERVAL = float(UI_CFG.get('spinner_interval', 0.3))

# === DERIVED CONSTANTS ======================================================
WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)
FREQ_RESOLUTION = SAMPLE_RATE / FFT_SIZE
# ============================================================================

class FrequencyMonitor:
    def __init__(self):
        self.ser = None
        self.audio_buffer = deque(maxlen=WINDOW_SIZE_SAMPLES)
        self.data_queue = queue.Queue()
        self.running = False
        
        # Setup logging and matplotlib for real-time plotting
        setup_logging(BASE / "app.log")
        logging.info("Starting frequency monitoring")
        logging.info(f"Serial={SERIAL_PORT} @ {BAUD_RATE} | SR={SAMPLE_RATE} | Block={BLOCK_SIZE}")

        # Setup matplotlib for real-time plotting
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Real-time Audio Frequency Analysis', fontsize=14)
        
        # Time domain plot
        self.ax1.set_title('Time Domain Signal')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True, alpha=0.3)
        
        # Frequency domain plot
        self.ax2.set_title('Frequency Spectrum (FFT)')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('dB SPL')
        self.ax2.set_xlim(FREQ_RANGE)
        self.ax2.grid(True, alpha=0.3)
        
        # Initialize plot lines
        self.time_line, = self.ax1.plot([], [], 'b-', linewidth=0.8)
        self.freq_line, = self.ax2.plot([], [], 'r-', linewidth=1.0)
        
        # Add text for dominant frequencies
        self.freq_text = self.ax2.text(0.02, 0.95, '', transform=self.ax2.transAxes,
                                      verticalalignment='top', bbox=dict(boxstyle='round', 
                                      facecolor='wheat', alpha=0.8))

    def connect_serial(self):
        """Connect to serial port using shared open/retry helper"""
        try:
            with (spinner("Connecting to serial...", interval_seconds=SPINNER_INTERVAL) if SPINNER_ENABLED else nullcontext()):
                self.ser = open_serial_with_retry(SERIAL_PORT, BAUD_RATE)
            if not self.ser:
                logging.error("Could not open serial port %s", SERIAL_PORT)
                return False
            try:
                self.ser.flushInput()
            except Exception:
                pass
            logging.info("Serial connected")
            return True
        except Exception:
            logging.exception("Critical: Could not open serial port %s", SERIAL_PORT)
            return False

    def read_serial_data(self):
        """Thread function to continuously read serial data with header sync"""
        logging.info("Starting frequency domain monitoring (header sync)")
        
        packet_size = BLOCK_SIZE * SAMPLE_WIDTH  # 1024
        checksum_size = 1
        total_packet_size = packet_size + checksum_size
        wait_stop_event = None
        wait_thread = None
        first_packet_received = False
        if SPINNER_ENABLED:
            wait_stop_event, wait_thread = start_spinner("Waiting for data...", interval_seconds=SPINNER_INTERVAL)
        while self.running:
            try:
                # 1) หา header 0xAA55
                # อ่านทีละไบต์จนเจอ 0xAA 0x55 ติดต่อกัน
                b = self.ser.read(1)
                if not b or b[0] != 0xAA:
                    continue
                b2 = self.ser.read(1)
                if not b2 or b2[0] != 0x55:
                    continue
                if not first_packet_received and SPINNER_ENABLED:
                    first_packet_received = True
                    stop_spinner(wait_stop_event, wait_thread)

                # 2) อ่าน PCM data ตามขนาด packet_size
                raw = self.ser.read(packet_size)
                if len(raw) < packet_size:
                    # อ่านไม่ครบ → ข้าม แล้ว sync ใหม่
                    continue

                # 3) แปลงเป็น int16 array
                samples = np.frombuffer(raw, dtype=np.int16)
                samples = samples / 32768.0
                # 4) เก็บลง buffer และคิว
                self.audio_buffer.extend(samples)
                if len(self.audio_buffer) >= WINDOW_SIZE_SAMPLES:
                    buffer_copy = np.array(self.audio_buffer)
                    self.data_queue.put(buffer_copy)

            except Exception as e:
                logging.exception("Error reading serial data")
                break
        if SPINNER_ENABLED:
            stop_spinner(wait_stop_event, wait_thread)


    def analyze_frequency(self, audio_data):
        """Perform frequency domain analysis"""
        # Apply windowing to reduce spectral leakage
        windowed_data = audio_data * signal.windows.hann(len(audio_data))
        
        # Perform FFT
        fft = np.fft.rfft(windowed_data, FFT_SIZE)
        fft_magnitude = np.abs(fft[:FFT_SIZE//2])
        
        # Convert to dB raw scale
        fft_db_raw = 20 * np.log10(fft_magnitude / 1000)  # Add small value to avoid log(0)
        
        # Convert to dB SPL using calibration formula
        fft_db_spl = 1.0497 * fft_db_raw + 120.1897 -79 
        
        # Create frequency axis
        freqs = np.fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)[:FFT_SIZE//2]
        
        # Find dominant frequencies
        # Only consider frequencies within our display range
        freq_mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
        masked_fft_spl = fft_db_spl.copy()
        masked_fft_spl[~freq_mask] = 0  # Suppress out-of-range frequencies
        
        # Find peaks (using a reasonable threshold for dB SPL values)
        peak_threshold = np.max(masked_fft_spl) - 20  # 20 dB below peak
        peaks, properties = signal.find_peaks(masked_fft_spl, height=peak_threshold, distance=20)
        
        # Get top 5 dominant frequencies
        if len(peaks) > 0:
            peak_heights = masked_fft_spl[peaks]
            top_indices = np.argsort(peak_heights)[-5:]  # Top 5
            top_peaks = peaks[top_indices]
            dominant_freqs = [(freqs[peak], masked_fft_spl[peak]) for peak in top_peaks]
            dominant_freqs.sort(key=lambda x: x[1], reverse=True)  # Sort by magnitude
        else:
            dominant_freqs = []
        
        return freqs, fft_db_spl, dominant_freqs

    def update_plot(self, frame):
        """Update plot with new data"""
        try:
            # Get latest data from queue
            audio_data = None
            while not self.data_queue.empty():
                audio_data = self.data_queue.get_nowait()
            
            if audio_data is None:
                return self.time_line, self.freq_line
            
            # Update time domain plot
            time_axis = np.arange(len(audio_data)) / SAMPLE_RATE
            self.time_line.set_data(time_axis, audio_data)
            self.ax1.set_xlim(0, len(audio_data) / SAMPLE_RATE)
            self.ax1.set_ylim(np.min(audio_data), np.max(audio_data))
            
            # Frequency analysis
            freqs, fft_db_spl, dominant_freqs = self.analyze_frequency(audio_data)
            
            # Update frequency domain plot
            freq_mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
            display_freqs = freqs[freq_mask]
            display_fft = fft_db_spl[freq_mask]
            
            self.freq_line.set_data(display_freqs, display_fft)
            
            if len(display_fft) > 0:
                self.ax2.set_ylim(np.min(display_fft) - 5, np.max(display_fft) + 5)
            
            # Update dominant frequencies text
            if dominant_freqs:
                freq_text = "Dominant Frequencies (Hz):\n"
                for i, (freq, spl_level) in enumerate(dominant_freqs[:3]):  # Show top 3
                    freq_text += f"{i+1}. {freq:.1f} Hz ({spl_level:.1f} dB SPL)\n"
            else:
                freq_text = "No dominant frequencies detected"
            
            self.freq_text.set_text(freq_text)
            
            return self.time_line, self.freq_line
            
        except Exception as e:
            logging.exception("Error updating plot")
            return self.time_line, self.freq_line

    def start_monitoring(self):
        """Start the frequency monitoring"""
        if not self.connect_serial():
            return
        
        self.running = True
        
        # Start serial reading thread
        serial_thread = threading.Thread(target=self.read_serial_data)
        serial_thread.daemon = True
        serial_thread.start()
        
        # Start animation
        try:
            ani = FuncAnimation(self.fig, self.update_plot, interval=UPDATE_INTERVAL, 
                              blit=False, cache_frame_data=False)
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            logging.info("User pressed Ctrl+C.")
        except Exception as e:
            logging.exception("Plot error")
        finally:
            self.stop_monitoring()

    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        logging.info("Stopping frequency monitoring...")
        self.running = False
        
        if self.ser and self.ser.is_open:
            self.ser.close()
            logging.info("Serial port closed.")
        
        logging.info("Program finished.")

def main():
    """Main function"""
    monitor = FrequencyMonitor()
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        logging.info("User interrupted.")
    except Exception:
        logging.exception("Unexpected error")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()