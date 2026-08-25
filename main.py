"""
Main Pipeline Orchestrator — Ambient Context Engine

Orchestrates all layers of the pipeline:
1. Starts screen capture and audio capture threads
2. Runs processing loops (OCR, Whisper, embeddings) in worker threads
3. Writes processed results to the database
4. Handles graceful shutdown on Ctrl+C
5. Runs periodic retention cleanup

Usage:
    python main.py                  # Start the full pipeline
    python main.py --no-audio       # Screen-only mode (no microphone)
    python main.py --no-screen      # Audio-only mode
"""

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from queue import Queue, Empty

from config import (
    CAPTURE_INTERVAL_SECS, AUDIO_CHUNK_SECS,
    RETENTION_DAYS, LOG_LEVEL,
)

logger = logging.getLogger("ace")


class PipelineOrchestrator:
    """
    Central orchestrator that wires up all pipeline components and
    manages their lifecycle.
    """

    def __init__(self, enable_screen: bool = True, enable_audio: bool = True):
        self.enable_screen = enable_screen
        self.enable_audio = enable_audio

        # Shared stop signal for all threads
        self.stop_event = threading.Event()

        # Queues for inter-thread communication
        self.screen_queue: Queue = Queue(maxsize=50)
        self.audio_queue: Queue = Queue(maxsize=20)

        # Components (initialized lazily in start())
        self.db = None
        self.embedder = None
        self.ocr_processor = None
        self.transcriber = None
        self.redactor = None
        self.enable_audio = enable_audio
        
        self.web_thread = None
        self.screen_thread = None
        self.audio_thread = None

        # Stats
        self._start_time = None
        self._screen_processed = 0
        self._audio_processed = 0

    def _init_components(self):
        """Initialize all pipeline components. Heavy models load here."""
        logger.info("=" * 60)
        logger.info("  Ambient Context Engine — Initializing")
        logger.info("=" * 60)

        # Storage
        logger.info("[1/4] Initializing database...")
        from storage.db import DatabaseManager
        self.db = DatabaseManager()

        # Embeddings
        logger.info("[2/4] Loading embedding model...")
        from processing.embed import EmbeddingGenerator
        self.embedder = EmbeddingGenerator()

        # OCR
        if self.enable_screen:
            logger.info("[3/4] Initializing OCR processor...")
            from processing.ocr import OCRProcessor
            self.ocr_processor = OCRProcessor()
        else:
            logger.info("[3/4] Screen capture disabled — skipping OCR")

        # Whisper
        if self.enable_audio:
            logger.info("[4/4] Loading Whisper model...")
            from processing.transcribe import Transcriber
            self.transcriber = Transcriber()
        else:
            logger.info("[4/4] Audio capture disabled — skipping Whisper")

        # Redactor
        from processing.redact import PIIRedactor
        self.redactor = PIIRedactor()

        # Start Web UI
        logger.info("[2/4] Starting Web UI server...")
        from interface.app import start_server
        self.web_thread = start_server(self.db, self.embedder, port=5000)

        # Start Capture Threads
        logger.info("[3/4] Starting Capture Threads...")
        """Start the background capture threads."""
        if self.enable_screen:
            from capture.screen_capture import ScreenCaptureThread
            self.screen_thread = ScreenCaptureThread(
                output_queue=self.screen_queue,
                stop_event=self.stop_event,
            )
            self.screen_thread.start()
            logger.info(f"Screen capture started (every {CAPTURE_INTERVAL_SECS}s)")

        if self.enable_audio:
            from capture.audio_capture import AudioCaptureThread
            self.audio_thread = AudioCaptureThread(
                output_queue=self.audio_queue,
                stop_event=self.stop_event,
            )
            self.audio_thread.start()
            logger.info(f"Audio capture started ({AUDIO_CHUNK_SECS}s chunks)")

    def _process_screen_event(self, event):
        """Process a single screen capture event: OCR → embed → store."""
        try:
            # Run OCR (includes deduplication)
            ocr_result = self.ocr_processor.process(
                frame=event.frame,
                timestamp=event.timestamp,
                window_title=event.window_title,
                process_name=event.process_name,
            )

            if ocr_result is None:
                return  # Skipped (empty or duplicate)

            # Redact PII
            redacted_text = self.redactor.redact(ocr_result.text)

            # Generate embedding
            embedding = self.embedder.generate(redacted_text)

            # Store in database
            self.db.insert_capture(
                source="screen",
                text_content=redacted_text,
                embedding=embedding,
                window_title=ocr_result.window_title,
                process_name=ocr_result.process_name,
                timestamp=ocr_result.timestamp,
            )

            self._screen_processed += 1
            logger.debug(
                f"Screen capture stored: {len(ocr_result.text)} chars "
                f"from '{ocr_result.window_title}' "
                f"(OCR: {ocr_result.ocr_time_ms:.0f}ms)"
            )

        except Exception as e:
            logger.error(f"Error processing screen event: {e}", exc_info=True)

    def _process_audio_chunk(self, chunk):
        """Process a single audio chunk: transcribe → embed → store."""
        try:
            # Run Whisper transcription
            transcript = self.transcriber.transcribe(
                audio_data=chunk.audio_data,
                timestamp_start=chunk.timestamp_start,
                timestamp_end=chunk.timestamp_end,
                duration_secs=chunk.duration_secs,
            )

            if transcript is None:
                return  # No speech detected

            # Redact PII
            redacted_text = self.redactor.redact(transcript.text)

            # Generate embedding
            embedding = self.embedder.generate(redacted_text)

            # Store in database
            self.db.insert_capture(
                source="audio",
                text_content=redacted_text,
                embedding=embedding,
                confidence=transcript.confidence,
                duration_secs=transcript.duration_secs,
                timestamp=transcript.timestamp_start,
            )

            self._audio_processed += 1
            logger.debug(
                f"Audio transcript stored: {len(transcript.text)} chars "
                f"(confidence={transcript.confidence:.2f}, "
                f"transcribe={transcript.transcribe_time_ms:.0f}ms)"
            )

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)

    def _processing_loop(self):
        """
        Main processing loop. Reads from both queues and processes events.
        Runs in the main thread.
        """
        logger.info("\n" + "=" * 60)
        logger.info("  Pipeline is now running — Press Ctrl+C to stop")
        logger.info("=" * 60 + "\n")

        last_stats_time = time.time()
        last_cleanup_time = time.time()
        STATS_INTERVAL = 60        # Print stats every 60 seconds
        CLEANUP_INTERVAL = 3600    # Run cleanup every hour

        while not self.stop_event.is_set():
            processed_any = False

            # Process screen captures
            if self.enable_screen:
                try:
                    event = self.screen_queue.get(timeout=0.1)
                    self._process_screen_event(event)
                    processed_any = True
                except Empty:
                    pass

            # Process audio chunks
            if self.enable_audio:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    self._process_audio_chunk(chunk)
                    processed_any = True
                except Empty:
                    pass

            # Periodic stats logging
            now = time.time()
            if now - last_stats_time > STATS_INTERVAL:
                self._log_stats()
                last_stats_time = now

            # Periodic retention cleanup
            if RETENTION_DAYS > 0 and now - last_cleanup_time > CLEANUP_INTERVAL:
                self.db.cleanup_old_captures()
                last_cleanup_time = now

            # Small sleep if nothing was processed to avoid busy-waiting
            if not processed_any:
                time.sleep(0.05)

    def _log_stats(self):
        """Log pipeline statistics."""
        elapsed = time.time() - self._start_time
        elapsed_min = elapsed / 60

        stats = self.db.get_stats()
        screen_stats = self.screen_thread.stats if self.screen_thread else {}
        audio_stats = self.audio_thread.stats if self.audio_thread else {}

        logger.info(
            f"📊 Pipeline stats ({elapsed_min:.0f} min): "
            f"screen={self._screen_processed} stored "
            f"({screen_stats.get('captured', 0)} captured, "
            f"{screen_stats.get('skipped', 0)} skipped) | "
            f"audio={self._audio_processed} stored "
            f"({audio_stats.get('chunks_captured', 0)} chunks, "
            f"{audio_stats.get('silent_skipped', 0)} silent) | "
            f"DB={stats['db_size_mb']:.1f}MB"
        )

    def start(self):
        """Start the full pipeline."""
        self._start_time = time.time()

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            logger.info("\n\nShutdown signal received...")
            self.stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Initialize all components and start capture threads
            self._init_components()

            # Run the processing loop (blocks until stop_event is set)
            self._processing_loop()

        except KeyboardInterrupt:
            logger.info("\nKeyboard interrupt received")
            self.stop_event.set()

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.stop_event.set()

        finally:
            self._shutdown()

    def _shutdown(self):
        """Gracefully shut down all components."""
        logger.info("Shutting down pipeline...")

        self.stop_event.set()

        # Wait for capture threads to finish
        if self.screen_thread and self.screen_thread.is_alive():
            self.screen_thread.join(timeout=5)
            logger.info("Screen capture thread stopped")

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=5)
            logger.info("Audio capture thread stopped")

        # Final stats
        if self._start_time:
            self._log_stats()

        # Close database
        if self.db:
            self.db.close()

        logger.info("Pipeline shutdown complete.")


# ─── Entry Point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ambient Context Engine — Local privacy-first screen & audio indexer"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Disable audio capture (screen-only mode)"
    )
    parser.add_argument(
        "--no-screen", action="store_true",
        help="Disable screen capture (audio-only mode)"
    )
    parser.add_argument(
        "--log-level", default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s │ %(levelname)-7s │ %(name)-15s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.no_audio and args.no_screen:
        print("Error: Cannot disable both audio and screen capture!")
        sys.exit(1)

    orchestrator = PipelineOrchestrator(
        enable_screen=not args.no_screen,
        enable_audio=not args.no_audio,
    )
    orchestrator.start()


if __name__ == "__main__":
    main()
