-- Ambient Context Engine — Database Schema
-- SQLite with FTS5 (full-text search) and sqlite-vec (vector search)

-- ─── Core Captures Table ────────────────────────────────────────────
-- Every piece of captured content (OCR text or audio transcript) is a row.
CREATE TABLE IF NOT EXISTS captures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,        -- ISO 8601 (e.g. 2026-08-10T14:30:00)
    source          TEXT    NOT NULL,        -- 'screen' or 'audio'
    text_content    TEXT    NOT NULL,        -- extracted OCR text or transcript
    window_title    TEXT,                    -- active window title (screen captures only)
    process_name    TEXT,                    -- active process name (screen captures only)
    confidence      REAL,                   -- transcription confidence (audio only)
    duration_secs   REAL                    -- chunk duration in seconds (audio only)
);

-- Index for fast time-range queries (session detection, summaries, retention cleanup)
CREATE INDEX IF NOT EXISTS idx_captures_timestamp ON captures(timestamp);
CREATE INDEX IF NOT EXISTS idx_captures_source    ON captures(source);

-- ─── Full-Text Search Index ─────────────────────────────────────────
-- FTS5 content-sync table for exact keyword matching.
-- Uses content= to avoid data duplication — FTS reads from captures table.
CREATE VIRTUAL TABLE IF NOT EXISTS captures_fts USING fts5(
    text_content,
    content=captures,
    content_rowid=id
);

-- Triggers to keep FTS5 in sync with the captures table
CREATE TRIGGER IF NOT EXISTS captures_fts_insert AFTER INSERT ON captures BEGIN
    INSERT INTO captures_fts(rowid, text_content)
    VALUES (new.id, new.text_content);
END;

CREATE TRIGGER IF NOT EXISTS captures_fts_delete AFTER DELETE ON captures BEGIN
    INSERT INTO captures_fts(captures_fts, rowid, text_content)
    VALUES ('delete', old.id, old.text_content);
END;

CREATE TRIGGER IF NOT EXISTS captures_fts_update AFTER UPDATE ON captures BEGIN
    INSERT INTO captures_fts(captures_fts, rowid, text_content)
    VALUES ('delete', old.id, old.text_content);
    INSERT INTO captures_fts(rowid, text_content)
    VALUES (new.id, new.text_content);
END;

-- ─── Vector Search Index ────────────────────────────────────────────
-- sqlite-vec virtual table for semantic similarity search.
-- 384-dim float32 vectors matching all-MiniLM-L6-v2 output.
-- rowid corresponds 1:1 with captures.id.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_captures USING vec0(
    embedding float[384]
);

-- ─── Sessions Table (Phase 2) ───────────────────────────────────────
-- Groups of temporally/contextually related captures.
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time  TEXT    NOT NULL,
    end_time    TEXT    NOT NULL,
    label       TEXT,                       -- dominant window/task label
    summary     TEXT                        -- LLM-generated session summary
);

CREATE INDEX IF NOT EXISTS idx_sessions_time ON sessions(start_time, end_time);

-- ─── Session Membership (Phase 2) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS session_captures (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    capture_id  INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    PRIMARY KEY (session_id, capture_id)
);

-- ─── Pipeline Metadata ─────────────────────────────────────────────
-- Tracks pipeline health and stats for the CLI `status` command.
CREATE TABLE IF NOT EXISTS pipeline_stats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    stat_key    TEXT    NOT NULL,            -- e.g. 'captures_total', 'ocr_avg_ms'
    stat_value  REAL    NOT NULL
);

-- ─── Context Graph (Phase 3) ────────────────────────────────────────
-- Represents links between captures (e.g. audio spoken during screen view)
CREATE TABLE IF NOT EXISTS graph_edges (
    source_id       INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    target_id       INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    relation_type   TEXT    NOT NULL,        -- 'OCCURRED_DURING', 'SAME_APP', etc.
    PRIMARY KEY (source_id, target_id, relation_type)
);
-- ─── Proactive Nudges (Phase 3) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS nudges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    nudge_text  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_graph_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_target ON graph_edges(target_id);
