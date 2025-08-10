"""Utility helpers for lightweight CLI spinner/indicator.

Exports a legacy-compatible `spinner_task(stop_event)` plus higher-level helpers:
- `start_spinner(...)` / `stop_spinner(...)`
- `spinner(...)` context manager
"""

import sys
import time
import threading
from contextlib import contextmanager
from threading import Event, Thread
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "spinner_task",
    "spinner_loop",
    "start_spinner",
    "stop_spinner",
    "spinner",
]

IO_LOCK = threading.RLock()


def spinner_loop(
    stop_event: Event,
    message: str = "Initializing fault detection...",
    interval_seconds: float = 0.2,
    frames: Optional[Sequence[str]] = None,
) -> None:
    """
    Core spinner loop. Prints a minimal spinner while `stop_event` is not set.

    - stop_event: signal to terminate the loop
    - message: left prefix text
    - interval_seconds: delay between frames
    - frames: custom spinner frames (defaults to | / - \\)
    """
    spinner_frames: Sequence[str] = list(frames) if frames else ["|", "/", "-", "\\"]
    frame_index = 0
    is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

    try:
        while not stop_event.is_set():
            if is_tty:
                with IO_LOCK:
                    print(
                        f"\r{message} {spinner_frames[frame_index]}",
                        end="",
                        flush=True,
                        file=sys.stdout,
                    )
            else:
                # In non-TTY, avoid carriage return noise. Emit heartbeat occasionally.
                if frame_index % max(1, int(round(1.0 / max(0.001, interval_seconds)))) == 0:
                    with IO_LOCK:
                        print(f"{message} ...", flush=True, file=sys.stdout)

            frame_index = (frame_index + 1) % len(spinner_frames)
            time.sleep(interval_seconds)
    finally:
        if is_tty:
            # Clear the line to avoid leaving spinner residue
            clear_len = len(message) + 4
            with IO_LOCK:
                print("\r" + (" " * clear_len), end="", flush=True, file=sys.stdout)
                print("\r", end="", flush=True, file=sys.stdout)


def spinner_task(stop_event: Event) -> None:
    """
    Backward-compatible wrapper used by existing callers.
    Displays a simple spinner with default message and interval.
    """
    spinner_loop(stop_event)


def start_spinner(
    message: str = "Initializing fault detection...",
    *,
    interval_seconds: float = 0.2,
    frames: Optional[Sequence[str]] = None,
) -> Tuple[Event, Thread]:
    """
    Start spinner in a daemon thread. Returns (stop_event, thread).
    Call `stop_spinner(stop_event, thread)` to stop and join.
    """
    stop_event = Event()
    thread = Thread(
        target=spinner_loop,
        args=(stop_event, message, interval_seconds, frames),
        daemon=True,
        name="cli-spinner",
    )
    thread.start()
    return stop_event, thread


def stop_spinner(stop_event: Event, thread: Optional[Thread] = None, *, timeout_seconds: float = 2.0) -> None:
    """Signal spinner to stop and optionally join the thread."""
    stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout_seconds)


@contextmanager
def spinner(
    message: str = "Initializing fault detection...",
    *,
    interval_seconds: float = 0.2,
    frames: Optional[Sequence[str]] = None,
):
    """
    Context manager that runs a spinner while inside the `with` block.
    Example:
        with spinner("Loading models..."):
            load_models()
    """
    stop_event, thread = start_spinner(message, interval_seconds=interval_seconds, frames=frames)
    try:
        yield
    finally:
        stop_spinner(stop_event, thread)
