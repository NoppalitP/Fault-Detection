import time
import logging
import serial

def open_serial_with_retry(port: str, baud_rate: int, retries: int = 5, delay: int = 3):
    for attempt in range(1, retries + 1):
        try:
            ser = serial.Serial(port, baud_rate, timeout=1)
            logging.info(f"Serial port opened successfully on attempt {attempt}")
            return ser
        except serial.SerialException as e:
            logging.warning(f"Attempt {attempt}/{retries} - Failed to open serial port: {e}")
            time.sleep(delay)
    logging.error("Exceeded maximum retry attempts to open serial port.")
    return None
