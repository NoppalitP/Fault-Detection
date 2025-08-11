import json
import numpy as np
import time
import threading
from collections import deque

P_REF = 20e-6  # 20 µPa
MIC_SENS_DBFS_AT_94 = -22.0   # SPM1423: -22 dBFS @ 94 dB SPL
BASE_DB = 94.0 - MIC_SENS_DBFS_AT_94  # 116 dB → dB SPL = dBFS + 116 + calib_offset
CAL_FILE = "spl_calib.json"
USE_A_WEIGHTING = True  # ให้สเปกตรัม/A-weight รวมสอดคล้องเครื่องวัดที่ตั้ง A
WINDOW_SIZE_SAMPLES = 1024  # Default window size for audio processing

def _float_pcm(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float32) / 32768.0   # int16 → [-1,1]
    return x.astype(np.float32, copy=False)

def a_weighting_db(f: np.ndarray) -> np.ndarray:
    # IEC 61672 (approx.) A-weighting in dB
    f2 = f * f
    ra_num = (12194.0**2) * (f2**2)
    ra_den = (f2 + 20.6**2) * (f2 + 12194.0**2) * np.sqrt((f2 + 107.7**2)*(f2 + 737.9**2))
    A = 2.0 + 20.0*np.log10(np.maximum(ra_num/np.maximum(ra_den, 1e-30), 1e-30))
    A[np.isnan(A)] = -np.inf
    return A

def dbfs_from_rms(y: np.ndarray) -> float:
    rms = np.sqrt(np.mean(y**2) + 1e-20)
    return 20.0*np.log10(rms)

class SPLCalibrator:
    def __init__(self, buffer_size=8192):
        self.calib_offset_db = 0.0
        self.audio_buffer = deque(maxlen=buffer_size)  # Use deque for efficient buffer management
        self.is_collecting = False
        self.collection_thread = None
        self.current_spl = 0.0
        self.readings_history = []
        
        # Load calibration offset from file
        try:
            with open(CAL_FILE, "r", encoding="utf-8") as f:
                self.calib_offset_db = float(json.load(f).get("calib_offset_db", 0.0))
                print(f"📏 Loaded calibration offset: {self.calib_offset_db:+.2f} dB")
        except Exception:
            print("ℹ️ No calibration file yet; using 0.0 dB offset.")

    def add_audio_data(self, audio_data):
        """Add audio data to the buffer for processing"""
        if isinstance(audio_data, (list, np.ndarray)):
            self.audio_buffer.extend(audio_data)
        else:
            self.audio_buffer.append(audio_data)

    def get_current_spl(self):
        """Get current SPL reading"""
        if len(self.audio_buffer) >= WINDOW_SIZE_SAMPLES:
            y = _float_pcm(np.array(list(self.audio_buffer)[-WINDOW_SIZE_SAMPLES:]))
            dbfs = dbfs_from_rms(y)
            self.current_spl = dbfs + BASE_DB + self.calib_offset_db
            return self.current_spl
        return None

    def start_continuous_monitoring(self):
        """Start continuous SPL monitoring in background thread"""
        if self.is_collecting:
            print("⚠️ Already collecting data!")
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._monitor_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        print("🎤 Started continuous SPL monitoring...")

    def stop_continuous_monitoring(self):
        """Stop continuous SPL monitoring"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)
        print("⏹️ Stopped SPL monitoring.")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_collecting:
            spl = self.get_current_spl()
            if spl is not None:
                self.readings_history.append({
                    'timestamp': time.time(),
                    'spl': spl,
                    'offset': self.calib_offset_db
                })
                # Keep only last 1000 readings
                if len(self.readings_history) > 1000:
                    self.readings_history = self.readings_history[-1000:]
            time.sleep(0.1)  # 10 Hz update rate

    def quick_calibration(self, ref_db_spl: float, duration: float = 10.0):
        """
        Quick calibration with automatic data collection
        """
        print(f"🧪 Quick calibration: Collecting {duration}s of data to match {ref_db_spl:.1f} dB SPL...")
        print("📊 Starting data collection...")
        
        # Start monitoring
        self.start_continuous_monitoring()
        time.sleep(duration)
        self.stop_continuous_monitoring()
        
        # Process collected data
        if not self.readings_history:
            print("⚠️ No data collected!")
            return False
        
        # Calculate median from collected readings
        spl_readings = [r['spl'] for r in self.readings_history]
        measured = float(np.median(spl_readings))
        
        # Calculate statistics
        std_dev = float(np.std(spl_readings))
        min_val = float(np.min(spl_readings))
        max_val = float(np.max(spl_readings))
        
        print(f"📈 Collected {len(spl_readings)} readings:")
        print(f"   Median: {measured:.2f} dB SPL")
        print(f"   Std Dev: {std_dev:.2f} dB")
        print(f"   Range: {min_val:.2f} - {max_val:.2f} dB")
        
        # Calculate and apply offset
        needed_offset = ref_db_spl - measured
        self.calib_offset_db += needed_offset
        
        # Save calibration
        with open(CAL_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "calib_offset_db": self.calib_offset_db,
                "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reference_spl": ref_db_spl,
                "measured_spl": measured,
                "offset_applied": needed_offset,
                "readings_count": len(spl_readings)
            }, f, indent=2)
        
        print(f"✅ Calibration complete!")
        print(f"   New offset: {self.calib_offset_db:+.2f} dB")
        print(f"   Applied correction: {needed_offset:+.2f} dB")
        print(f"   Saved to: {CAL_FILE}")
        
        return True

    def manual_calibration(self, ref_db_spl: float, seconds: float = 5.0):
        """
        Manual calibration with user control
        """
        print(f"🧪 Manual calibration: {seconds}s to match {ref_db_spl:.1f} dB SPL...")
        print("🎯 Press Enter when ready to start...")
        input()
        
        print("⏱️ Starting in 3...")
        time.sleep(1)
        print("⏱️ Starting in 2...")
        time.sleep(1)
        print("⏱️ Starting in 1...")
        time.sleep(1)
        print("🎤 Collecting data...")
        
        t_end = time.time() + seconds
        readings = []
        
        while time.time() < t_end:
            spl = self.get_current_spl()
            if spl is not None:
                readings.append(spl)
                print(f"\r📊 Current: {spl:.1f} dB SPL | Readings: {len(readings)}", end="")
            time.sleep(0.1)
        
        print()  # New line after progress
        
        if not readings:
            print("⚠️ Not enough samples for calibration.")
            return False

        measured = float(np.median(readings))
        needed_offset = ref_db_spl - measured
        self.calib_offset_db += needed_offset

        with open(CAL_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "calib_offset_db": self.calib_offset_db,
                "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reference_spl": ref_db_spl,
                "measured_spl": measured,
                "offset_applied": needed_offset,
                "readings_count": len(readings)
            }, f, indent=2)
        
        print(f"✅ Calibration complete!")
        print(f"   New offset: {self.calib_offset_db:+.2f} dB (Δ {needed_offset:+.2f} dB)")
        print(f"   Saved to: {CAL_FILE}")
        
        return True

    def get_calibration_info(self):
        """Get current calibration information"""
        try:
            with open(CAL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print("📋 Calibration Information:")
                print(f"   Offset: {data.get('calib_offset_db', 0):+.2f} dB")
                print(f"   Date: {data.get('calibration_date', 'Unknown')}")
                print(f"   Reference SPL: {data.get('reference_spl', 'Unknown')} dB")
                print(f"   Measured SPL: {data.get('measured_spl', 'Unknown')} dB")
                return data
        except Exception:
            print("ℹ️ No calibration file found.")
            return None

    def reset_calibration(self):
        """Reset calibration to zero"""
        self.calib_offset_db = 0.0
        with open(CAL_FILE, "w", encoding="utf-8") as f:
            json.dump({"calib_offset_db": 0.0}, f)
        print("🔄 Calibration reset to 0.0 dB")

    def monitor_and_collect(self, duration=None):
        """
        Real-time monitoring with data collection
        Press Ctrl+C to stop
        """
        print("🎤 Real-time SPL Monitoring & Data Collection")
        print("=" * 50)
        print("Press Ctrl+C to stop monitoring")
        print("=" * 50)
        
        self.start_continuous_monitoring()
        start_time = time.time()
        
        try:
            while True:
                spl = self.get_current_spl()
                if spl is not None:
                    elapsed = time.time() - start_time
                    readings_count = len(self.readings_history)
                    print(f"\r📊 SPL: {spl:6.1f} dB | Readings: {readings_count:4d} | Time: {elapsed:6.1f}s", end="", flush=True)
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    print(f"\n⏰ Duration limit reached ({duration}s)")
                    break
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
        
        finally:
            self.stop_continuous_monitoring()
            self._save_collected_data()
            self._show_collection_summary()

    def _save_collected_data(self):
        """Save collected data to file"""
        if not self.readings_history:
            print("⚠️ No data to save!")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"spl_data_{timestamp}.json"
        
        data_to_save = {
            "collection_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_readings": len(self.readings_history),
                "calibration_offset": self.calib_offset_db
            },
            "readings": self.readings_history,
            "statistics": self._calculate_statistics()
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"💾 Data saved to: {filename}")

    def _calculate_statistics(self):
        """Calculate statistics from collected data"""
        if not self.readings_history:
            return {}
        
        spl_values = [r['spl'] for r in self.readings_history]
        
        return {
            "mean_spl": float(np.mean(spl_values)),
            "median_spl": float(np.median(spl_values)),
            "std_dev_spl": float(np.std(spl_values)),
            "min_spl": float(np.min(spl_values)),
            "max_spl": float(np.max(spl_values))
        }

    def _show_collection_summary(self):
        """Show summary of collected data"""
        if not self.readings_history:
            print("📊 No data collected")
            return
        
        stats = self._calculate_statistics()
        
        print("\n📊 Collection Summary:")
        print("=" * 30)
        print(f"Readings: {len(self.readings_history)}")
        print(f"Mean SPL: {stats['mean_spl']:.1f} dB")
        print(f"Median SPL: {stats['median_spl']:.1f} dB")
        print(f"Std Dev: {stats['std_dev_spl']:.1f} dB")
        print(f"Range: {stats['min_spl']:.1f} - {stats['max_spl']:.1f} dB")
        print(f"Calibration Offset: {self.calib_offset_db:+.2f} dB")

    def simple_monitor(self, update_interval=1.0):
        """
        Simple continuous monitoring without data collection
        Press Ctrl+C to stop
        """
        print("🎤 Simple SPL Monitoring (Press Ctrl+C to stop)")
        print("=" * 40)
        
        try:
            while True:
                spl = self.get_current_spl()
                if spl is not None:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] Current SPL: {spl:6.1f} dB")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped")

    def collect_data_for_duration(self, duration_seconds, filename=None):
        """
        Collect data for a specific duration and save to file
        """
        print(f"📊 Collecting data for {duration_seconds} seconds...")
        
        self.start_continuous_monitoring()
        time.sleep(duration_seconds)
        self.stop_continuous_monitoring()
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"spl_collection_{duration_seconds}s_{timestamp}.json"
        
        self._save_collected_data()
        self._show_collection_summary()
        
        return filename

    def monitor_and_collect(self, duration=None, save_to_file=True):
        """
        Real-time monitoring with data collection
        Press 'q' to quit, 's' to save current data, 'c' to clear history
        """
        print("🎤 Real-time SPL Monitoring & Data Collection")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit monitoring")
        print("  's' - Save current data to file")
        print("  'c' - Clear data history")
        print("  'i' - Show current calibration info")
        print("  'r' - Reset calibration")
        print("=" * 50)
        
        self.start_continuous_monitoring()
        
        start_time = time.time()
        last_display = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Update display every 0.5 seconds
                if current_time - last_display >= 0.5:
                    spl = self.get_current_spl()
                    if spl is not None:
                        # Clear line and show current reading
                        print(f"\r📊 Current SPL: {spl:6.1f} dB | Readings: {len(self.readings_history):4d} | Time: {current_time - start_time:6.1f}s", end="", flush=True)
                    last_display = current_time
                
                # Check for user input (non-blocking)
                if self._check_key_press():
                    key = input("\nEnter command (q/s/c/i/r): ").lower().strip()
                    
                    if key == 'q':
                        print("\n⏹️ Stopping monitoring...")
                        break
                    elif key == 's':
                        self._save_collected_data()
                    elif key == 'c':
                        self.readings_history.clear()
                        print("🗑️ Data history cleared!")
                    elif key == 'i':
                        self.get_calibration_info()
                    elif key == 'r':
                        self.reset_calibration()
                    else:
                        print("❓ Unknown command. Use: q/s/c/i/r")
                
                # Check duration limit
                if duration and (current_time - start_time) >= duration:
                    print(f"\n⏰ Duration limit reached ({duration}s)")
                    break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
        
        finally:
            self.stop_continuous_monitoring()
            
            # Auto-save if we have data
            if save_to_file and self.readings_history:
                self._save_collected_data()
            
            # Show summary
            self._show_collection_summary()

    def _check_key_press(self):
        """Simple non-blocking key check (Windows compatible)"""
        try:
            import msvcrt
            return msvcrt.kbhit()
        except ImportError:
            # For non-Windows systems, we'll use a different approach
            return False

    def _save_collected_data(self):
        """Save collected data to file"""
        if not self.readings_history:
            print("⚠️ No data to save!")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"spl_data_{timestamp}.json"
        
        data_to_save = {
            "collection_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": self.readings_history[-1]['timestamp'] - self.readings_history[0]['timestamp'],
                "total_readings": len(self.readings_history),
                "calibration_offset": self.calib_offset_db
            },
            "readings": self.readings_history,
            "statistics": self._calculate_statistics()
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"💾 Data saved to: {filename}")

    def _calculate_statistics(self):
        """Calculate statistics from collected data"""
        if not self.readings_history:
            return {}
        
        spl_values = [r['spl'] for r in self.readings_history]
        
        return {
            "mean_spl": float(np.mean(spl_values)),
            "median_spl": float(np.median(spl_values)),
            "std_dev_spl": float(np.std(spl_values)),
            "min_spl": float(np.min(spl_values)),
            "max_spl": float(np.max(spl_values)),
            "range_spl": float(np.max(spl_values) - np.min(spl_values))
        }

    def _show_collection_summary(self):
        """Show summary of collected data"""
        if not self.readings_history:
            print("📊 No data collected")
            return
        
        stats = self._calculate_statistics()
        duration = self.readings_history[-1]['timestamp'] - self.readings_history[0]['timestamp']
        
        print("\n📊 Collection Summary:")
        print("=" * 30)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Readings: {len(self.readings_history)}")
        print(f"Mean SPL: {stats['mean_spl']:.1f} dB")
        print(f"Median SPL: {stats['median_spl']:.1f} dB")
        print(f"Std Dev: {stats['std_dev_spl']:.1f} dB")
        print(f"Range: {stats['min_spl']:.1f} - {stats['max_spl']:.1f} dB")
        print(f"Calibration Offset: {self.calib_offset_db:+.2f} dB")

    def simple_monitor(self, update_interval=1.0):
        """
        Simple continuous monitoring without data collection
        Press Ctrl+C to stop
        """
        print("🎤 Simple SPL Monitoring (Press Ctrl+C to stop)")
        print("=" * 40)
        
        try:
            while True:
                spl = self.get_current_spl()
                if spl is not None:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] Current SPL: {spl:6.1f} dB")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped")

    def collect_data_for_duration(self, duration_seconds, filename=None):
        """
        Collect data for a specific duration and save to file
        """
        print(f"📊 Collecting data for {duration_seconds} seconds...")
        
        self.start_continuous_monitoring()
        time.sleep(duration_seconds)
        self.stop_continuous_monitoring()
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"spl_collection_{duration_seconds}s_{timestamp}.json"
        
        self._save_collected_data()
        self._show_collection_summary()
        
        return filename

# Example usage functions
def demo_calibration():
    """Demo function showing how to use the calibrator"""
    calibrator = SPLCalibrator()
    
    print("🎯 SPL Meter Calibration Demo")
    print("=" * 40)
    
    # Show current calibration
    calibrator.get_calibration_info()
    
    # Quick calibration example
    print("\n🚀 Quick Calibration Example:")
    print("1. Set your reference SPL meter to 94 dB")
    print("2. Place both microphones in the same position")
    print("3. Run the calibration")
    
    ref_spl = float(input("Enter reference SPL (e.g., 94.0): ") or "94.0")
    duration = float(input("Enter collection duration in seconds (e.g., 10): ") or "10.0")
    
    # Simulate some audio data (replace with real audio input)
    print("🎤 Simulating audio data...")
    for i in range(100):
        # Simulate audio samples (replace with real audio input)
        fake_audio = np.random.normal(0, 0.1, 1024)
        calibrator.add_audio_data(fake_audio)
        time.sleep(0.1)
    
    # Run calibration
    success = calibrator.quick_calibration(ref_spl, duration)
    
    if success:
        print("\n✅ Calibration successful!")
        calibrator.get_calibration_info()
    else:
        print("\n❌ Calibration failed!")

def demo_monitoring():
    """Demo function showing monitoring and data collection"""
    calibrator = SPLCalibrator()
    
    print("📊 SPL Monitoring & Data Collection Demo")
    print("=" * 50)
    
    print("Choose monitoring mode:")
    print("1. Simple monitoring (just watch dB levels)")
    print("2. Data collection with monitoring")
    print("3. Collect data for specific duration")
    
    choice = input("Enter choice (1-3): ").strip()
    
    # Simulate audio data (replace with real audio input)
    print("🎤 Simulating audio data...")
    
    if choice == "1":
        # Simple monitoring
        print("Starting simple monitoring...")
        # In real usage, you would add audio data continuously here
        for i in range(50):
            fake_audio = np.random.normal(0, 0.1, 1024)
            calibrator.add_audio_data(fake_audio)
            time.sleep(0.1)
        calibrator.simple_monitor(update_interval=1.0)
        
    elif choice == "2":
        # Data collection with monitoring
        print("Starting data collection monitoring...")
        # In real usage, you would add audio data continuously here
        for i in range(50):
            fake_audio = np.random.normal(0, 0.1, 1024)
            calibrator.add_audio_data(fake_audio)
            time.sleep(0.1)
        calibrator.monitor_and_collect(duration=30)  # 30 seconds
        
    elif choice == "3":
        # Collect for specific duration
        duration = float(input("Enter duration in seconds: ") or "10")
        print(f"Collecting data for {duration} seconds...")
        # In real usage, you would add audio data continuously here
        for i in range(50):
            fake_audio = np.random.normal(0, 0.1, 1024)
            calibrator.add_audio_data(fake_audio)
            time.sleep(0.1)
        filename = calibrator.collect_data_for_duration(duration)
        print(f"Data saved to: {filename}")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Calibration demo")
    print("2. Monitoring demo")
    
    mode = input("Enter choice (1-2): ").strip()
    
    if mode == "1":
        demo_calibration()
    elif mode == "2":
        demo_monitoring()
    else:
        print("Invalid choice!")