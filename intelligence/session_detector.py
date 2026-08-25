"""
Intelligence Layer — Session Detector

Infers user task "sessions" from raw captures based on temporal proximity
and window title continuity.

A session is a continuous block of activity focused on a specific task
or context. A new session is started if:
1. There is an idle gap (no captures) for more than 5 minutes.
2. The active window/process changes significantly.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from storage.db import DatabaseManager

logger = logging.getLogger(__name__)

IDLE_GAP_MINUTES = 5

class SessionDetector:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def _calculate_gap_minutes(self, time1_str: str, time2_str: str) -> float:
        """Calculate the difference in minutes between two ISO 8601 strings."""
        try:
            t1 = datetime.fromisoformat(time1_str)
            t2 = datetime.fromisoformat(time2_str)
            diff = abs((t2 - t1).total_seconds()) / 60.0
            return diff
        except Exception:
            return 0.0

    def build_sessions(self, start_time: str, end_time: str) -> int:
        """
        Group captures in the given time range into sessions and store them
        in the database. Returns the number of sessions created.
        """
        captures = self.db.get_captures_in_range(start_time, end_time)
        if not captures:
            logger.info("No captures found in the specified range to build sessions.")
            return 0

        sessions = []
        current_session = {
            "start_time": captures[0]["timestamp"],
            "end_time": captures[0]["timestamp"],
            "label": captures[0].get("window_title") or captures[0].get("process_name") or "Unknown Task",
            "capture_ids": [captures[0]["id"]],
            "window_title": captures[0].get("window_title"),
            "process_name": captures[0].get("process_name")
        }

        for i in range(1, len(captures)):
            capture = captures[i]
            prev_capture = captures[i-1]

            gap_mins = self._calculate_gap_minutes(prev_capture["timestamp"], capture["timestamp"])
            window_changed = (capture.get("window_title") != current_session["window_title"])

            # Determine if we should start a new session
            # 1. Significant idle gap
            # 2. Window title changed (could be more sophisticated, e.g., only if changed for more than a few captures, but keeping simple for MVP)
            is_new_session = (gap_mins > IDLE_GAP_MINUTES) or window_changed

            if is_new_session:
                # Save current session
                sessions.append(current_session)
                
                # Start new session
                current_session = {
                    "start_time": capture["timestamp"],
                    "end_time": capture["timestamp"],
                    "label": capture.get("window_title") or capture.get("process_name") or "Unknown Task",
                    "capture_ids": [capture["id"]],
                    "window_title": capture.get("window_title"),
                    "process_name": capture.get("process_name")
                }
            else:
                # Add to current session
                current_session["end_time"] = capture["timestamp"]
                current_session["capture_ids"].append(capture["id"])

        # Append the last session
        if current_session:
            sessions.append(current_session)

        # Store sessions in the database
        saved_count = 0
        try:
            for sess in sessions:
                # Create session record
                cursor = self.db.conn.execute(
                    """
                    INSERT INTO sessions (start_time, end_time, label)
                    VALUES (?, ?, ?)
                    """,
                    (sess["start_time"], sess["end_time"], sess["label"])
                )
                session_id = cursor.lastrowid

                # Create session_captures links
                for cap_id in sess["capture_ids"]:
                    self.db.conn.execute(
                        "INSERT INTO session_captures (session_id, capture_id) VALUES (?, ?)",
                        (session_id, cap_id)
                    )
                saved_count += 1
            self.db.conn.commit()
            logger.info(f"Successfully built and saved {saved_count} sessions.")
        except Exception as e:
            self.db.conn.rollback()
            logger.error(f"Error saving sessions to database: {e}", exc_info=True)
            raise

        return saved_count
