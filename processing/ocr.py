"""
Processing Layer — OCR

Extracts text from screen capture frames using Tesseract OCR.
Includes pre-processing (grayscale, thresholding, DPI scaling) for
better recognition, and deduplication to avoid storing identical text
when the screen content hasn't meaningfully changed.
"""

import logging
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

import cv2
import numpy as np
import pytesseract

from config import (
    TESSERACT_CMD, TESSERACT_LANG, TESSERACT_PSM, TESSERACT_OEM,
    OCR_DEDUP_SIMILARITY, OCR_DEDUP_HISTORY,
)

logger = logging.getLogger(__name__)

# Point pytesseract to the Tesseract binary
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


@dataclass
class OCRResult:
    """Result of OCR processing on a single frame."""
    timestamp: str
    text: str
    window_title: str
    process_name: str
    ocr_time_ms: float              # How long OCR took (for monitoring)


class OCRProcessor:
    """
    Processes screen capture frames through Tesseract OCR with
    pre-processing and deduplication.
    """

    def __init__(self):
        self._recent_texts: deque[str] = deque(maxlen=OCR_DEDUP_HISTORY)
        self._total_processed = 0
        self._total_deduped = 0

        # Tesseract config string
        self._tess_config = (
            f"--psm {TESSERACT_PSM} "
            f"--oem {TESSERACT_OEM} "
            f"-l {TESSERACT_LANG}"
        )

        # Verify Tesseract is available
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR version: {version}")
        except Exception as e:
            logger.error(
                f"Tesseract not found at '{TESSERACT_CMD}'. "
                f"Please install Tesseract and update TESSERACT_CMD in config.py. "
                f"Error: {e}"
            )
            raise

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Pre-process a BGR frame for optimal OCR recognition.

        Steps:
        1. Convert to grayscale
        2. Scale up if too small (Tesseract works best at ~300 DPI)
        3. Apply adaptive thresholding for better contrast
        4. Denoise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Scale up small frames (Tesseract prefers ~300 DPI)
        height, width = gray.shape
        if width < 1920:
            scale = 1920 / width
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        # Apply adaptive thresholding for better text/background contrast
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # Light denoising (removes small artifacts without blurring text)
        processed = cv2.medianBlur(processed, 3)

        return processed

    def _is_duplicate(self, text: str) -> bool:
        """
        Check if the extracted text is too similar to recently captured
        texts, using SequenceMatcher for fuzzy matching.
        """
        for recent in self._recent_texts:
            ratio = SequenceMatcher(None, text, recent).ratio()
            if ratio > OCR_DEDUP_SIMILARITY:
                return True
        return False

    def _clean_text(self, raw_text: str) -> str:
        """
        Clean OCR output: strip whitespace, remove empty lines,
        collapse excessive whitespace.
        """
        lines = raw_text.strip().splitlines()
        # Remove empty or whitespace-only lines
        cleaned = [line.strip() for line in lines if line.strip()]
        return "\n".join(cleaned)

    def process(
        self,
        frame: np.ndarray,
        timestamp: str,
        window_title: str,
        process_name: str,
    ) -> Optional[OCRResult]:
        """
        Run OCR on a captured frame. Returns an OCRResult if meaningful
        text was extracted and it's not a duplicate. Returns None if
        the frame should be skipped.
        """
        import time
        start_time = time.time()

        try:
            # Pre-process the frame
            processed = self.preprocess_frame(frame)

            # Run Tesseract OCR
            raw_text = pytesseract.image_to_string(
                processed, config=self._tess_config
            )

            # Clean the output
            text = self._clean_text(raw_text)

            elapsed_ms = (time.time() - start_time) * 1000
            self._total_processed += 1

            # Skip empty results
            if not text or len(text) < 10:
                logger.debug(
                    f"OCR produced insufficient text ({len(text)} chars), skipping"
                )
                return None

            # Skip duplicates
            if self._is_duplicate(text):
                self._total_deduped += 1
                logger.debug("OCR text is duplicate of recent capture, skipping")
                return None

            # Store in recent history for dedup
            self._recent_texts.append(text)

            logger.debug(
                f"OCR completed in {elapsed_ms:.0f}ms: "
                f"{len(text)} chars from '{window_title}'"
            )

            return OCRResult(
                timestamp=timestamp,
                text=text,
                window_title=window_title,
                process_name=process_name,
                ocr_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"OCR processing error: {e}", exc_info=True)
            return None

    @property
    def stats(self) -> dict:
        return {
            "total_processed": self._total_processed,
            "total_deduped": self._total_deduped,
        }


# ─── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import mss

    logging.basicConfig(level=logging.DEBUG)

    print("Testing OCR processor on current screen...")
    processor = OCRProcessor()

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        frame = np.array(raw)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    from datetime import datetime
    result = processor.process(
        frame_bgr,
        timestamp=datetime.now().isoformat(),
        window_title="Test Window",
        process_name="test.exe",
    )

    if result:
        print(f"\n  OCR Time: {result.ocr_time_ms:.0f}ms")
        print(f"  Text length: {len(result.text)} chars")
        print(f"  First 200 chars:\n  {result.text[:200]}")
    else:
        print("  No meaningful text extracted from screen")

    print(f"\n  Stats: {processor.stats}")
    print("OCR test complete!")
