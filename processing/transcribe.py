"""
Processing Layer — Speech-to-Text Transcription

Uses openai-whisper for CPU-based transcription. Processes buffered audio
chunks from the capture layer.

Switched from faster-whisper to openai-whisper to avoid the PyAV (av)
DLL dependency that may be blocked by Windows Application Control policies.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    WHISPER_MODEL, WHISPER_LANGUAGE,
)

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    """Result of transcribing an audio chunk."""
    timestamp_start: str
    timestamp_end: str
    text: str
    language: str
    confidence: float
    duration_secs: float
    transcribe_time_ms: float       # Processing time (for monitoring)


class Transcriber:
    """
    Speech-to-text transcription using openai-whisper on CPU.

    The Whisper model is loaded once at initialization and reused
    for all subsequent transcriptions.
    """

    def __init__(self):
        logger.info(f"Loading Whisper model '{WHISPER_MODEL}' (device=cpu)...")

        load_start = time.time()

        import whisper
        self.model = whisper.load_model(WHISPER_MODEL)

        load_time = time.time() - load_start
        logger.info(f"Whisper model loaded in {load_time:.1f}s")

        self._total_transcribed = 0
        self._total_duration_secs = 0.0

    def transcribe(
        self,
        audio_data: np.ndarray,
        timestamp_start: str,
        timestamp_end: str,
        duration_secs: float,
    ) -> Optional[TranscriptResult]:
        """
        Transcribe an audio chunk. Returns a TranscriptResult if speech
        was detected, or None if the chunk is silent/unintelligible.

        Args:
            audio_data: float32 mono audio at 16kHz
            timestamp_start: ISO 8601 timestamp when recording started
            timestamp_end: ISO 8601 timestamp when recording ended
            duration_secs: Duration of the audio chunk in seconds
        """
        start_time = time.time()

        try:
            # Ensure correct dtype — whisper expects float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Run transcription
            # openai-whisper's transcribe() accepts a numpy array directly
            result = self.model.transcribe(
                audio_data,
                language=WHISPER_LANGUAGE,
                fp16=False,     # CPU-only: must disable fp16
                verbose=False,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            full_text = result.get("text", "").strip()

            # Skip if no speech was detected
            if not full_text:
                logger.debug(
                    f"Whisper: no speech detected in {duration_secs}s chunk "
                    f"({elapsed_ms:.0f}ms processing)"
                )
                return None

            detected_lang = result.get("language", WHISPER_LANGUAGE)

            # openai-whisper doesn't expose segment-level log-probs easily,
            # so we use a fixed moderate confidence value.
            confidence_score = 0.75

            self._total_transcribed += 1
            self._total_duration_secs += duration_secs

            logger.debug(
                f"Whisper: transcribed {duration_secs}s → {len(full_text)} chars "
                f"in {elapsed_ms:.0f}ms"
            )

            return TranscriptResult(
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                text=full_text,
                language=detected_lang,
                confidence=confidence_score,
                duration_secs=duration_secs,
                transcribe_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None

    @property
    def stats(self) -> dict:
        return {
            "total_transcribed": self._total_transcribed,
            "total_audio_duration_secs": round(self._total_duration_secs, 1),
        }


# ─── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("Testing Whisper transcriber...")
    transcriber = Transcriber()

    from datetime import datetime
    sample_rate = 16000
    duration = 5  # seconds
    silent_audio = np.zeros(sample_rate * duration, dtype=np.float32)

    result = transcriber.transcribe(
        audio_data=silent_audio,
        timestamp_start=datetime.now().isoformat(),
        timestamp_end=datetime.now().isoformat(),
        duration_secs=duration,
    )

    if result:
        print(f"  Transcript: '{result.text}'")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Time: {result.transcribe_time_ms:.0f}ms")
    else:
        print("  No speech detected in silent test audio (expected)")

    print(f"  Stats: {transcriber.stats}")
    print("Transcriber test complete!")
