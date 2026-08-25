"""
Intelligence Layer — Summarizer

Generates daily/weekly summaries of the user's activity.
Retrieves sessions for a time period, formats them, and uses the local LLM
to generate a cohesive narrative "story".
"""

import logging
from datetime import datetime, timedelta
import requests

from storage.db import DatabaseManager
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """You are a personal productivity assistant. Your task is to generate a concise, engaging summary of the user's activity based on the provided session logs.
Write a narrative "story" of how the user spent their time. Group related tasks, highlight main areas of focus, and mention significant apps or contexts used.
Keep it readable, professional, and insightful. Do not just list the sessions."""

SUMMARY_TEMPLATE = """Here is the log of the user's sessions for the requested period:

{session_logs}

Please write a summary narrative of this activity."""

class Summarizer:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def _get_sessions_in_range(self, start_time: str, end_time: str) -> list[dict]:
        """Fetch sessions from the database for the given time range."""
        rows = self.db.conn.execute(
            """
            SELECT id, start_time, end_time, label, summary
            FROM sessions
            WHERE start_time >= ? AND start_time <= ?
            ORDER BY start_time ASC
            """,
            (start_time, end_time)
        ).fetchall()
        return [dict(row) for row in rows]

    def _format_sessions(self, sessions: list[dict]) -> str:
        """Format session list into a string for the LLM prompt."""
        if not sessions:
            return "No activity recorded."
            
        lines = []
        for s in sessions:
            try:
                st = datetime.fromisoformat(s['start_time']).strftime('%H:%M')
                et = datetime.fromisoformat(s['end_time']).strftime('%H:%M')
                lines.append(f"- [{st} - {et}] {s['label']}")
            except Exception:
                lines.append(f"- {s['label']}")
        
        return "\n".join(lines)

    def _generate_llm_summary(self, session_text: str) -> str:
        """Call Ollama to generate the narrative summary."""
        prompt = SUMMARY_TEMPLATE.format(session_logs=session_text)
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": SUMMARY_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "top_p": 0.9,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Failed to generate summary with Ollama: {e}")
            return "⚠️ Error generating summary: Could not reach LLM."

    def summarize_period(self, start_time: str, end_time: str) -> str:
        """Generate a summary for a specific time range."""
        sessions = self._get_sessions_in_range(start_time, end_time)
        if not sessions:
            return "No activity found for this period."
            
        session_text = self._format_sessions(sessions)
        
        logger.info(f"Generating summary based on {len(sessions)} sessions...")
        summary_text = self._generate_llm_summary(session_text)
        
        return summary_text

    def summarize_today(self) -> str:
        """Convenience method to summarize today's activity."""
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        end = now.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()
        return self.summarize_period(start, end)
