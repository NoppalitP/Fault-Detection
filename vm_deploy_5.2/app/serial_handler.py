import time
import logging
import serial

def open_serial_with_retry(possible_ports, baud_rate: int, retries: int = 5, delay: int = 3):
    """
    Try to open a serial port from a list of possible ports with retries.

    Args:
        possible_ports (list[str]): List of possible port names, e.g., ["COM3", "COM5"].
        baud_rate (int): Baud rate for the serial connection.
        retries (int): Number of retry attempts per port.
        delay (int): Delay (seconds) between retries.

    Returns:
        serial.Serial or None: The opened serial object if successful, else None.
    """
    for port in possible_ports:
        for attempt in range(1, retries + 1):
            try:
                ser = serial.Serial(port, baud_rate, timeout=1)
                logging.info(f"Serial port {port} opened successfully on attempt {attempt}")
                return ser
            except serial.SerialException as e:
                logging.warning(f"Attempt {attempt}/{retries} - Failed to open {port}: {e}")
                time.sleep(delay)

    logging.error("Exceeded maximum retry attempts for all ports.")
    return None