"""
Storage Layer — Database Manager

Handles all SQLite operations: connection management, schema initialization,
insert/search/cleanup operations. Integrates FTS5 (keyword search) and
sqlite-vec (semantic vector search) with hybrid Reciprocal Rank Fusion.
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

import sqlite_vec
from config import (
    DB_PATH, EMBEDDING_DIM, SEARCH_TOP_K, RRF_K, RETENTION_DAYS
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe SQLite database manager with FTS5 + sqlite-vec support."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = self._connect()
        self._init_schema()
        logger.info(f"Database initialized at {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and sqlite-vec loaded."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row

        # Enable WAL mode for concurrent reads/writes
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache

        # Load sqlite-vec extension
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        logger.debug("sqlite-vec extension loaded successfully")
        return conn

    def _init_schema(self):
        """Run schema.sql to create tables if they don't exist."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, "r") as f:
            schema_sql = f.read()

        try:
            self.conn.executescript(schema_sql)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            self.conn.rollback()
            raise

    # ─── Insert Operations ───────────────────────────────────────────

    def insert_capture(
        self,
        source: str,
        text_content: str,
        embedding: np.ndarray,
        window_title: Optional[str] = None,
        process_name: Optional[str] = None,
        confidence: Optional[float] = None,
        duration_secs: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """
        Insert a capture event into all three stores (captures, FTS, vec)
        in a single transaction. Returns the inserted row ID.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
                f"got {embedding.shape}"
            )

        try:
            cursor = self.conn.execute(
                """
                INSERT INTO captures
                    (timestamp, source, text_content, window_title,
                     process_name, confidence, duration_secs)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (timestamp, source, text_content, window_title,
                 process_name, confidence, duration_secs),
            )
            row_id = cursor.lastrowid

            # Insert embedding into vec_captures (rowid must match captures.id)
            embedding_bytes = embedding.astype(np.float32).tobytes()
            self.conn.execute(
                "INSERT INTO vec_captures(rowid, embedding) VALUES (?, ?)",
                (row_id, embedding_bytes),
            )

            self.conn.commit()
            logger.debug(
                f"Inserted capture id={row_id} source={source} "
                f"text_len={len(text_content)}"
            )
            return row_id

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert capture: {e}")
            raise

    def insert_edge(self, source_id: int, target_id: int, relation_type: str):
        """Insert a relationship between two captures in the context graph."""
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO graph_edges (source_id, target_id, relation_type) VALUES (?, ?, ?)",
                (source_id, target_id, relation_type)
            )
            self.conn.commit()
            logger.debug(f"Inserted graph edge: {source_id} -> {target_id} ({relation_type})")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert graph edge: {e}")

    def insert_nudge(self, nudge_text: str):
        """Insert a proactive nudge."""
        try:
            self.conn.execute(
                "INSERT INTO nudges (timestamp, nudge_text) VALUES (?, ?)",
                (datetime.now().isoformat(), nudge_text)
            )
            self.conn.commit()
            logger.debug("Inserted new proactive nudge.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert nudge: {e}")

    # ─── Search Operations ───────────────────────────────────────────

    def search_semantic(
        self, query_embedding: np.ndarray, limit: int = SEARCH_TOP_K
    ) -> list[dict]:
        """
        K-nearest-neighbor search using sqlite-vec.
        Returns list of dicts with capture metadata + distance.
        """
        embedding_bytes = query_embedding.astype(np.float32).tobytes()

        rows = self.conn.execute(
            """
            SELECT
                c.id, c.timestamp, c.source, c.text_content,
                c.window_title, c.process_name, v.distance
            FROM vec_captures v
            JOIN captures c ON c.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (embedding_bytes, limit),
        ).fetchall()

        return [dict(row) for row in rows]

    def search_keyword(
        self, query_text: str, limit: int = SEARCH_TOP_K
    ) -> list[dict]:
        """
        Full-text keyword search using FTS5.
        Returns list of dicts with capture metadata + FTS rank.
        """
        rows = self.conn.execute(
            """
            SELECT
                c.id, c.timestamp, c.source, c.text_content,
                c.window_title, c.process_name,
                captures_fts.rank AS fts_rank
            FROM captures_fts
            JOIN captures c ON c.id = captures_fts.rowid
            WHERE captures_fts MATCH ?
            ORDER BY captures_fts.rank
            LIMIT ?
            """,
            (query_text, limit),
        ).fetchall()

        return [dict(row) for row in rows]

    def search_hybrid(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        limit: int = SEARCH_TOP_K,
    ) -> list[dict]:
        """
        Hybrid search combining semantic + keyword results using
        Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank_i)) across each ranker.
        """
        semantic_results = self.search_semantic(query_embedding, limit=limit * 2)
        keyword_results = self.search_keyword(query_text, limit=limit * 2)

        # Build RRF scores
        rrf_scores: dict[int, float] = {}
        result_map: dict[int, dict] = {}

        for rank, result in enumerate(semantic_results):
            rid = result["id"]
            rrf_scores[rid] = rrf_scores.get(rid, 0) + 1.0 / (RRF_K + rank + 1)
            result_map[rid] = result

        for rank, result in enumerate(keyword_results):
            rid = result["id"]
            rrf_scores[rid] = rrf_scores.get(rid, 0) + 1.0 / (RRF_K + rank + 1)
            result_map[rid] = result

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for rid in sorted_ids[:limit]:
            entry = result_map[rid].copy()
            entry["rrf_score"] = rrf_scores[rid]
            results.append(entry)

        return results

    # ─── Query Operations ────────────────────────────────────────────

    def get_captures_in_range(
        self, start_time: str, end_time: str
    ) -> list[dict]:
        """Retrieve all captures within a time range (ISO 8601 strings)."""
        rows = self.conn.execute(
            """
            SELECT id, timestamp, source, text_content,
                   window_title, process_name
            FROM captures
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
            """,
            (start_time, end_time),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_recent_captures(self, limit: int = 20) -> list[dict]:
        """Retrieve the most recent N captures."""
        rows = self.conn.execute(
            """
            SELECT id, timestamp, source, text_content,
                   window_title, process_name
            FROM captures
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_latest_nudge(self) -> Optional[dict]:
        """Get the most recent nudge."""
        row = self.conn.execute(
            "SELECT timestamp, nudge_text FROM nudges ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_stats(self) -> dict:
        """Return database statistics for the CLI status command."""
        total = self.conn.execute(
            "SELECT COUNT(*) FROM captures"
        ).fetchone()[0]

        screen_count = self.conn.execute(
            "SELECT COUNT(*) FROM captures WHERE source = 'screen'"
        ).fetchone()[0]

        audio_count = self.conn.execute(
            "SELECT COUNT(*) FROM captures WHERE source = 'audio'"
        ).fetchone()[0]

        # Database file size
        db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        db_size_mb = db_size_bytes / (1024 * 1024)

        # Time range
        first = self.conn.execute(
            "SELECT MIN(timestamp) FROM captures"
        ).fetchone()[0]

        last = self.conn.execute(
            "SELECT MAX(timestamp) FROM captures"
        ).fetchone()[0]

        return {
            "total_captures": total,
            "screen_captures": screen_count,
            "audio_captures": audio_count,
            "db_size_mb": round(db_size_mb, 2),
            "earliest_capture": first,
            "latest_capture": last,
        }

    # ─── Retention / Cleanup ─────────────────────────────────────────

    def cleanup_old_captures(self, retention_days: int = RETENTION_DAYS):
        """
        Delete captures older than retention_days.
        Also removes corresponding FTS and vec entries via triggers + manual vec cleanup.
        """
        if retention_days <= 0:
            return 0

        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()

        # Get IDs to delete (needed for vec_captures cleanup)
        ids_to_delete = self.conn.execute(
            "SELECT id FROM captures WHERE timestamp < ?", (cutoff,)
        ).fetchall()

        if not ids_to_delete:
            return 0

        id_list = [row[0] for row in ids_to_delete]

        # Delete from vec_captures (no trigger support for virtual tables)
        placeholders = ",".join("?" * len(id_list))
        self.conn.execute(
            f"DELETE FROM vec_captures WHERE rowid IN ({placeholders})",
            id_list,
        )

        # Delete from captures (FTS cleanup happens via trigger)
        deleted = self.conn.execute(
            "DELETE FROM captures WHERE timestamp < ?", (cutoff,)
        ).rowcount

        self.conn.commit()
        logger.info(
            f"Retention cleanup: deleted {deleted} captures older than "
            f"{retention_days} days"
        )
        return deleted

    # ─── Lifecycle ───────────────────────────────────────────────────

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ─── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    print("Testing DatabaseManager...")
    with DatabaseManager() as db:
        # Insert a test capture
        test_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        row_id = db.insert_capture(
            source="screen",
            text_content="This is a test OCR capture from VS Code editor",
            embedding=test_embedding,
            window_title="test.py - Visual Studio Code",
            process_name="Code.exe",
        )
        print(f"  Inserted test capture with id={row_id}")

        # Insert another
        test_embedding2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        row_id2 = db.insert_capture(
            source="audio",
            text_content="Let's discuss the project architecture tomorrow",
            embedding=test_embedding2,
            confidence=0.92,
            duration_secs=30.0,
        )
        print(f"  Inserted test audio capture with id={row_id2}")

        # Keyword search
        kw_results = db.search_keyword("project architecture")
        print(f"  Keyword search 'project architecture': {len(kw_results)} results")

        # Semantic search
        sem_results = db.search_semantic(test_embedding, limit=5)
        print(f"  Semantic search: {len(sem_results)} results")

        # Stats
        stats = db.get_stats()
        print(f"  Stats: {json.dumps(stats, indent=2)}")

        # Recent
        recent = db.get_recent_captures(5)
        print(f"  Recent captures: {len(recent)}")

    print("All database tests passed!")
