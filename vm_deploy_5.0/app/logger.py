import logging
import sys
from pathlib import Path
from typing import Any

try:
    # colorama provides cross-platform ANSI color support on Windows terminals
    from colorama import Fore, Style, Back, init as colorama_init
except Exception:  # pragma: no cover - optional dependency guard
    Fore = Style = None  # type: ignore
    def colorama_init(*args: Any, **kwargs: Any) -> None:  # type: ignore
        return


class _ColorFormatter(logging.Formatter):
    """Colorizes level names for console readability."""

    LEVEL_COLOR = {
        logging.DEBUG: ("DEBUG", ""),
        logging.INFO: ("INFO", ""),
        logging.WARNING: ("WARN", ""),
        logging.ERROR: ("ERROR", ""),
        logging.CRITICAL: ("CRIT", ""),
    }

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

        # If colorama is available, set color mappings
        if Fore and Style:
            self.LEVEL_COLOR = {
                logging.DEBUG: (f"{Style.DIM}DEBUG{Style.RESET_ALL}", Style.DIM),
                logging.INFO: (f"{Fore.GREEN}INFO{Style.RESET_ALL}", ""),
                logging.WARNING: (f"{Fore.YELLOW}WARN{Style.RESET_ALL}", ""),
                logging.ERROR: (f"{Fore.RED}ERROR{Style.RESET_ALL}", ""),
                logging.CRITICAL: (f"{Fore.WHITE}{Style.BRIGHT}{Back.RED}CRIT{Style.RESET_ALL}", ""),
            }

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        levelname = record.levelname
        mapped = self.LEVEL_COLOR.get(record.levelno)
        if mapped:
            colored_level, prefix_style = mapped
            record.levelname = colored_level
        try:
            return super().format(record)
        finally:
            record.levelname = levelname


def setup_logging(log_file: Path) -> None:
    """Configure file + colorized console logging.

    - File: full timestamp, plain text
    - Console: short timestamp, colored levels (if supported)
    """

    # Initialize color on Windows terminals
    colorama_init(autoreset=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Reset existing handlers to avoid duplicate logs if re-initialized
    root_logger.handlers.clear()

    # File handler (plain)
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (colorized)
    # Send console logs to stderr to avoid interleaving with spinner (stdout)
    console_handler = logging.StreamHandler(sys.stderr)
    # Reuse the same I/O lock as spinner to serialize writes across streams
    try:
        from app.utils import IO_LOCK  # type: ignore
        console_handler.lock = IO_LOCK  # type: ignore[attr-defined]
    except Exception:
        pass
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_ColorFormatter())

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
