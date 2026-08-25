"""
Intelligence Layer — Proactive Nudges

Evaluates recent user activity to surface helpful, proactive insights.
For example, detecting long unbroken focus sessions, repetitive errors,
or suggesting a break.
"""

import logging
from datetime import datetime, timedelta
import requests

from storage.db import DatabaseManager
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

NUDGE_SYSTEM_PROMPT = """You are a proactive, helpful AI productivity coach observing the user's recent screen and audio activity.
Your goal is to provide ONE short, insightful "nudge" or observation.
Examples:
- "You've been focused on VS Code for 3 hours straight. Might be a good time for a short break."
- "You've been researching Python asyncio for a while. Want me to summarize the best practices?"
- "You just had a 45-minute meeting about the Q3 roadmap. Should I generate action items?"

Keep the nudge under 2 sentences. Be friendly and helpful, not annoying. If there is no clear pattern or nothing worth noting, reply exactly with: "NO_NUDGE"."""

NUDGE_TEMPLATE = """Here is the user's activity for the past {hours} hours:

{activity_log}

Based on this, what is your proactive nudge? (Remember, reply NO_NUDGE if nothing stands out)."""

class NudgeAgent:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def evaluate_recent_activity(self, hours: float = 2.0) -> str:
        """
        Look at the last `hours` of activity and generate a nudge.
        Returns the nudge string, or None if no nudge is needed.
        """
        now = datetime.now()
        start_time = (now - timedelta(hours=hours)).isoformat()
        end_time = now.isoformat()

        # Fetch recent sessions rather than raw captures to save LLM context
        rows = self.db.conn.execute(
            """
            SELECT start_time, end_time, label
            FROM sessions
            WHERE start_time >= ? AND start_time <= ?
            ORDER BY start_time ASC
            """,
            (start_time, end_time)
        ).fetchall()

        if not rows:
            return None

        # Format activity
        lines = []
        for r in rows:
            st = datetime.fromisoformat(r['start_time']).strftime('%H:%M')
            et = datetime.fromisoformat(r['end_time']).strftime('%H:%M')
            lines.append(f"[{st} - {et}] Task: {r['label']}")

        activity_text = "\n".join(lines)
        prompt = NUDGE_TEMPLATE.format(hours=hours, activity_log=activity_text)

        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": NUDGE_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "top_p": 0.9,
                    },
                },
                timeout=60,
            )
            resp.raise_for_status()
            response_text = resp.json().get("response", "").strip()

            if "NO_NUDGE" in response_text.upper():
                return None
            
            return response_text

        except Exception as e:
            logger.error(f"Failed to generate nudge: {e}")
            return None

# ─── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    db = DatabaseManager()
    agent = NudgeAgent(db)
    
    # Needs some sessions in DB to work
    nudge = agent.evaluate_recent_activity(hours=4.0)
    print("Nudge:", nudge)
