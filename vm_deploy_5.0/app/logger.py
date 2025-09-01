import logging
import sys
from pathlib import Path
from typing import Any

from logging.handlers import TimedRotatingFileHandler
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
    - File: rotate every 1 day, keep 7 backups
    - Console: colored levels (if supported)
    """
    from colorama import init as colorama_init
    colorama_init(autoreset=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    # File handler (rotate daily, keep last 7 days)
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="D",            # D = day
        interval=1,          # rotate every 1 day
        backupCount=7,       # keep 7 files, delete older
        encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (colorized)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_ColorFormatter())

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
