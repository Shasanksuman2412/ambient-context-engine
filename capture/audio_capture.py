"""
Capture Layer — Audio Capture

Continuously records system/microphone audio at 16kHz mono (Whisper's
expected format), buffers it into configurable-length chunks, and pushes
AudioChunk instances into a shared queue for the transcription pipeline.

Runs in its own thread.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from typing import Optional

import numpy as np
import sounddevice as sd

from config import AUDIO_CHUNK_SECS, AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A buffered audio segment ready for transcription."""
    timestamp_start: str            # ISO 8601 — when recording started
    timestamp_end: str              # ISO 8601 — when recording ended
    audio_data: np.ndarray          # float32 mono, 16kHz
    duration_secs: float


def list_audio_devices() -> list[dict]:
    """List available audio input devices for user selection."""
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": dev["default_samplerate"],
            })
    return input_devices


class AudioCaptureThread(threading.Thread):
    """
    Background thread that records audio from the default input device
    (microphone) in fixed-length chunks and pushes them to a queue.

    Audio is captured at 16kHz mono float32 — the format expected by
    faster-whisper.
    """

    def __init__(
        self,
        output_queue: Queue,
        stop_event: threading.Event,
        chunk_duration: float = AUDIO_CHUNK_SECS,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        device: Optional[int] = None,
    ):
        super().__init__(daemon=True, name="AudioCapture")
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.device = device

        # State
        self._chunk_count = 0
        self._silent_count = 0

        # Pre-compute the number of samples per chunk
        self._samples_per_chunk = int(sample_rate * chunk_duration)

    def _is_silent(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Check if an audio chunk is effectively silent.
        Uses RMS energy threshold to avoid transcribing dead air.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        return rms < threshold

    def run(self):
        logger.info(
            f"Audio capture started: chunk={self.chunk_duration}s, "
            f"rate={self.sample_rate}Hz, device={self.device or 'default'}"
        )

        try:
            while not self.stop_event.is_set():
                try:
                    timestamp_start = datetime.now().isoformat()

                    # Record a chunk of audio (blocking call)
                    audio_data = sd.rec(
                        frames=self._samples_per_chunk,
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype="float32",
                        device=self.device,
                    )
                    sd.wait()  # Block until recording is complete

                    timestamp_end = datetime.now().isoformat()

                    # Flatten from (samples, 1) to (samples,)
                    audio_flat = audio_data.flatten()

                    # Skip silent chunks to save CPU on transcription
                    if self._is_silent(audio_flat):
                        self._silent_count += 1
                        if self._silent_count % 10 == 0:
                            logger.debug(
                                f"Audio: {self._silent_count} silent chunks skipped"
                            )
                        continue

                    chunk = AudioChunk(
                        timestamp_start=timestamp_start,
                        timestamp_end=timestamp_end,
                        audio_data=audio_flat,
                        duration_secs=self.chunk_duration,
                    )

                    self.output_queue.put(chunk)
                    self._chunk_count += 1

                    if self._chunk_count % 5 == 0:
                        logger.info(
                            f"Audio chunks captured: {self._chunk_count} "
                            f"(silent skipped: {self._silent_count})"
                        )

                except sd.PortAudioError as e:
                    logger.error(f"Audio device error: {e}")
                    # Wait before retrying on device errors
                    self.stop_event.wait(timeout=5.0)

                except Exception as e:
                    logger.error(f"Audio capture error: {e}", exc_info=True)
                    self.stop_event.wait(timeout=1.0)

        except Exception as e:
            logger.error(f"Audio capture thread fatal error: {e}", exc_info=True)

        logger.info(
            f"Audio capture stopped. Chunks: {self._chunk_count}, "
            f"Silent skipped: {self._silent_count}"
        )

    @property
    def stats(self) -> dict:
        return {
            "chunks_captured": self._chunk_count,
            "silent_skipped": self._silent_count,
        }


# ─── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("Available audio input devices:")
    for dev in list_audio_devices():
        print(f"  [{dev['index']}] {dev['name']} ({dev['channels']}ch)")

    print(f"\nRecording 2 chunks of {AUDIO_CHUNK_SECS}s each...")
    q = Queue()
    stop = threading.Event()

    thread = AudioCaptureThread(q, stop, chunk_duration=5)  # Short chunks for test
    thread.start()

    captured = 0
    while captured < 2:
        try:
            chunk = q.get(timeout=30)
            print(
                f"  Chunk #{captured + 1}: "
                f"samples={len(chunk.audio_data)}, "
                f"duration={chunk.duration_secs}s, "
                f"rms={np.sqrt(np.mean(chunk.audio_data ** 2)):.4f}"
            )
            captured += 1
        except Exception:
            print("  (Timeout or silent — no audio chunk received)")
            break

    stop.set()
    thread.join(timeout=10)
    print(f"Audio capture test complete! Stats: {thread.stats}")
