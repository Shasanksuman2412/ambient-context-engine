"""
Capture Layer — Screen Capture

Continuously captures the primary monitor at a configurable interval,
detects frame changes using OpenCV diffing, and records the active window
metadata (title + process name) via pywin32.

Runs in its own thread, pushing CaptureEvent instances into a shared queue.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from typing import Optional

import cv2
import mss
import numpy as np

# Windows-specific imports for active window detection
import win32gui
import win32process
import psutil

from config import CAPTURE_INTERVAL_SECS, FRAME_DIFF_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class CaptureEvent:
    """A single screen capture with metadata."""
    timestamp: str                  # ISO 8601
    frame: np.ndarray               # BGR numpy array (full resolution)
    window_title: str               # Active window title
    process_name: str               # Active process name (e.g. "Code.exe")


def get_active_window_info() -> tuple[str, str]:
    """
    Get the foreground window's title and process name.
    Returns ("Unknown", "Unknown") if detection fails.
    """
    try:
        hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(hwnd) or "Untitled"

        # Get the process ID from the window handle
        _, pid = win32process.GetWindowThreadProcessId(hwnd)

        try:
            process = psutil.Process(pid)
            process_name = process.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_name = "Unknown"

        return window_title, process_name

    except Exception as e:
        logger.debug(f"Could not detect active window: {e}")
        return "Unknown", "Unknown"


def compute_frame_diff(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    threshold: int = FRAME_DIFF_THRESHOLD,
) -> bool:
    """
    Compare two grayscale frames. Returns True if the frame changed
    significantly (more than `threshold` pixels differ).
    Returns True if prev_gray is None (first frame always counts).
    """
    if prev_gray is None:
        return True

    if prev_gray.shape != curr_gray.shape:
        return True

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(binary)

    logger.debug(f"Frame diff: {changed_pixels} changed pixels (threshold={threshold})")
    return changed_pixels > threshold


class ScreenCaptureThread(threading.Thread):
    """
    Background thread that captures the primary monitor at regular intervals,
    performs frame-diff detection, and pushes changed frames to a queue.
    """

    def __init__(
        self,
        output_queue: Queue,
        stop_event: threading.Event,
        interval: float = CAPTURE_INTERVAL_SECS,
        diff_threshold: int = FRAME_DIFF_THRESHOLD,
    ):
        super().__init__(daemon=True, name="ScreenCapture")
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.interval = interval
        self.diff_threshold = diff_threshold

        # State
        self._prev_gray: Optional[np.ndarray] = None
        self._capture_count = 0
        self._skip_count = 0

    def run(self):
        logger.info(
            f"Screen capture started: interval={self.interval}s, "
            f"diff_threshold={self.diff_threshold}px"
        )

        with mss.mss() as sct:
            # Use the primary monitor (index 1; index 0 is "all monitors")
            monitor = sct.monitors[1]
            logger.info(
                f"Capturing monitor: {monitor['width']}x{monitor['height']}"
            )

            while not self.stop_event.is_set():
                loop_start = time.time()

                try:
                    # Grab the screen
                    raw = sct.grab(monitor)
                    frame = np.array(raw)  # BGRA format

                    # Convert to grayscale for diff comparison
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

                    # Check if the frame actually changed
                    if compute_frame_diff(self._prev_gray, gray, self.diff_threshold):
                        # Convert BGRA → BGR for downstream processing
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                        # Get active window info
                        window_title, process_name = get_active_window_info()

                        event = CaptureEvent(
                            timestamp=datetime.now().isoformat(),
                            frame=frame_bgr,
                            window_title=window_title,
                            process_name=process_name,
                        )

                        self.output_queue.put(event)
                        self._capture_count += 1

                        if self._capture_count % 20 == 0:
                            logger.info(
                                f"Screen captures: {self._capture_count} captured, "
                                f"{self._skip_count} skipped (unchanged)"
                            )
                    else:
                        self._skip_count += 1

                    self._prev_gray = gray

                except Exception as e:
                    logger.error(f"Screen capture error: {e}", exc_info=True)

                # Sleep for the remainder of the interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.interval - elapsed)
                if sleep_time > 0:
                    self.stop_event.wait(timeout=sleep_time)

        logger.info(
            f"Screen capture stopped. Total: {self._capture_count} captured, "
            f"{self._skip_count} skipped"
        )

    @property
    def stats(self) -> dict:
        return {
            "captured": self._capture_count,
            "skipped": self._skip_count,
        }


# ─── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("Testing screen capture (5 captures, then stop)...")
    q = Queue()
    stop = threading.Event()

    thread = ScreenCaptureThread(q, stop, interval=1.0)
    thread.start()

    captured = 0
    while captured < 5:
        event = q.get(timeout=10)
        title, proc = get_active_window_info()
        print(
            f"  Capture #{captured + 1}: "
            f"frame={event.frame.shape}, "
            f"window='{event.window_title}', "
            f"process='{event.process_name}'"
        )
        captured += 1

    stop.set()
    thread.join(timeout=5)
    print(f"Screen capture test passed! Stats: {thread.stats}")
