"""
Interface Layer — Web UI Server

Flask-based REST API and HTML dashboard for viewing sessions, querying the
RAG agent, and displaying proactive nudges.
"""

import logging
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import threading

from storage.db import DatabaseManager
from intelligence.rag import RAGAgent
from intelligence.nudge_agent import NudgeAgent

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global instances initialized before running the server
db = None
rag_agent = None
nudge_agent = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sessions")
def get_sessions():
    """Returns the most recent N sessions."""
    try:
        limit = int(request.args.get("limit", 10))
        rows = db.conn.execute(
            """
            SELECT id, start_time, end_time, label
            FROM sessions
            ORDER BY start_time DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
        return jsonify([dict(row) for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recent")
def get_recent_captures():
    """Returns recent raw captures."""
    try:
        limit = int(request.args.get("limit", 20))
        captures = db.get_recent_captures(limit)
        return jsonify(captures)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """RAG agent query endpoint."""
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        result = rag_agent.query(question)
        return jsonify({
            "answer": result.get("answer", "No answer generated."),
            "sources": result.get("sources", []),
            "search_time_ms": result.get("search_time_ms", 0),
            "generate_time_ms": result.get("generate_time_ms", 0),
        })
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate answer"}), 500

@app.route("/api/nudge")
def get_nudge():
    """Generates or retrieves a proactive nudge."""
    try:
        # Check if we have a recent nudge stored
        latest = db.get_latest_nudge()
        # If it's less than 5 minutes old, return it to save LLM calls
        if latest:
            # simple check could go here, but for now just return the latest if we want to rely on the background thread
            pass
            
        # Or generate one on demand if requested
        nudge = nudge_agent.evaluate_recent_activity(hours=2.0)
        if nudge:
            db.insert_nudge(nudge)
            return jsonify({"nudge": nudge})
        else:
            return jsonify({"nudge": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats")
def get_stats():
    """Returns DB stats."""
    try:
        return jsonify(db.get_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def start_server(database: DatabaseManager, embedder, port: int = 5000):
    """Starts the Flask server in a background thread."""
    global db, rag_agent, nudge_agent
    db = database
    rag_agent = RAGAgent(db, embedder)
    nudge_agent = NudgeAgent(db)
    
    logger.info(f"Starting Web UI on http://localhost:{port}")
    # Run in a separate thread so it doesn't block the main pipeline
    thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True
    )
    thread.start()
    return thread

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_db = DatabaseManager()
    app.run(debug=True, port=5000)
