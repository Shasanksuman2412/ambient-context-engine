"""
Interface Layer — Command-Line Interface

Provides the user-facing CLI for searching, querying, and monitoring
the ambient context engine.

Commands:
    search <query>      — Hybrid search over captured context
    ask <question>      — RAG query (LLM-generated answer)
    summary [period]    — Generate a narrative summary of activity (e.g. today)
    recent [N]          — Show last N captured events
    status              — Pipeline and database stats
    cleanup             — Run retention cleanup manually
"""

import argparse
import sys
import logging
import json
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ─── ANSI Color Codes ────────────────────────────────────────────────
class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    RED     = "\033[31m"
    WHITE   = "\033[97m"


def _format_timestamp(iso_str: str) -> str:
    """Convert ISO timestamp to a human-friendly format."""
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now()
        diff = now - dt

        if diff.days == 0:
            return f"today at {dt.strftime('%I:%M %p')}"
        elif diff.days == 1:
            return f"yesterday at {dt.strftime('%I:%M %p')}"
        elif diff.days < 7:
            return dt.strftime("%A at %I:%M %p")
        else:
            return dt.strftime("%b %d at %I:%M %p")
    except (ValueError, TypeError):
        return iso_str or "unknown time"


def _source_icon(source: str) -> str:
    """Return an icon for the capture source."""
    return "🖥️ " if source == "screen" else "🎤"


def _print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}\n")


def _print_separator():
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")


# ─── Command Handlers ────────────────────────────────────────────────

def cmd_search(args, rag_agent):
    """Search captured context."""
    query = " ".join(args.query)
    if not query:
        print(f"{Colors.RED}Error: Please provide a search query{Colors.RESET}")
        return

    _print_header(f"Search: \"{query}\"")

    results = rag_agent.search_only(query, mode=args.mode, top_k=args.limit)

    if not results:
        print(f"{Colors.YELLOW}  No results found.{Colors.RESET}")
        return

    for i, r in enumerate(results, 1):
        source = _source_icon(r["source"])
        time_str = _format_timestamp(r["timestamp"])
        window = r.get("window_title", "")

        print(f"  {Colors.BOLD}{Colors.GREEN}[{i}]{Colors.RESET} {source} {Colors.DIM}{time_str}{Colors.RESET}")

        if window:
            print(f"      {Colors.BLUE}📌 {window}{Colors.RESET}")

        # Show text preview (first 150 chars)
        text = r["text_content"]
        preview = text[:150].replace("\n", " ")
        if len(text) > 150:
            preview += "..."
        print(f"      {preview}")

        if r.get("rrf_score"):
            print(f"      {Colors.DIM}relevance: {r['rrf_score']:.4f}{Colors.RESET}")

        _print_separator()

    print(f"\n{Colors.DIM}  {len(results)} results found{Colors.RESET}")


def cmd_ask(args, rag_agent):
    """Ask a question using RAG (LLM-generated answer)."""
    question = " ".join(args.question)
    if not question:
        print(f"{Colors.RED}Error: Please provide a question{Colors.RESET}")
        return

    _print_header(f"Question: \"{question}\"")
    print(f"  {Colors.DIM}Thinking...{Colors.RESET}", end="", flush=True)

    response = rag_agent.query(question)
    # Clear the "Thinking..." text
    print(f"\r{' ' * 40}\r", end="")

    # Print the answer
    print(f"  {Colors.BOLD}{Colors.WHITE}Answer:{Colors.RESET}")
    print()

    # Indent the answer
    for line in response["answer"].split("\n"):
        print(f"    {line}")
    print()

    # Print metadata
    print(
        f"  {Colors.DIM}"
        f"Search: {response['search_time_ms']:.0f}ms | "
        f"Generate: {response['generate_time_ms']:.0f}ms | "
        f"Sources: {len(response['sources'])}"
        f"{Colors.RESET}"
    )

    if args.show_sources and response["sources"]:
        print(f"\n  {Colors.BOLD}Sources:{Colors.RESET}")
        for i, s in enumerate(response["sources"][:5], 1):
            time_str = _format_timestamp(s["timestamp"])
            source = _source_icon(s["source"])
            print(f"    [{i}] {source} {time_str}")
            preview = s["text_content"][:80].replace("\n", " ")
            print(f"        {Colors.DIM}{preview}...{Colors.RESET}")


def cmd_recent(args, db):
    """Show recent captures."""
    limit = args.count
    _print_header(f"Recent {limit} Captures")

    results = db.get_recent_captures(limit=limit)

    if not results:
        print(f"{Colors.YELLOW}  No captures yet. Start the pipeline first.{Colors.RESET}")
        return

    for r in results:
        source = _source_icon(r["source"])
        time_str = _format_timestamp(r["timestamp"])
        window = r.get("window_title", "")

        print(f"  {source} {Colors.DIM}{time_str}{Colors.RESET}")
        if window:
            print(f"    {Colors.BLUE}📌 {window}{Colors.RESET}")

        text = r["text_content"]
        preview = text[:120].replace("\n", " ")
        if len(text) > 120:
            preview += "..."
        print(f"    {preview}")
        _print_separator()


def cmd_status(args, db, pipeline_status=None):
    """Show pipeline and database status."""
    _print_header("Engine Status")

    stats = db.get_stats()

    print(f"  {Colors.BOLD}Database:{Colors.RESET}")
    print(f"    📊 Total captures:  {stats['total_captures']}")
    print(f"    🖥️  Screen captures: {stats['screen_captures']}")
    print(f"    🎤 Audio captures:  {stats['audio_captures']}")
    print(f"    💾 Database size:   {stats['db_size_mb']:.2f} MB")
    print(f"    📅 First capture:   {_format_timestamp(stats['earliest_capture'])}")
    print(f"    📅 Last capture:    {_format_timestamp(stats['latest_capture'])}")

    if pipeline_status:
        print(f"\n  {Colors.BOLD}Pipeline:{Colors.RESET}")
        for key, value in pipeline_status.items():
            print(f"    {key}: {value}")


def cmd_summary(args, db):
    """Generate a summary of sessions."""
    from intelligence.session_detector import SessionDetector
    from intelligence.summarizer import Summarizer
    from datetime import datetime, timedelta

    _print_header(f"Summary: {args.period}")

    # First, ensure sessions are built up to now
    detector = SessionDetector(db)
    now = datetime.now()
    if args.period == "today":
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    elif args.period == "week":
        start = now - timedelta(days=now.weekday())
        start_time = start.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    else:
        print(f"{Colors.RED}Unsupported period. Use 'today' or 'week'.{Colors.RESET}")
        return
        
    end_time = now.isoformat()
    
    print(f"  {Colors.DIM}Building sessions...{Colors.RESET}")
    detector.build_sessions(start_time, end_time)

    print(f"  {Colors.DIM}Generating narrative summary...{Colors.RESET}")
    summarizer = Summarizer(db)
    
    if args.period == "today":
        summary = summarizer.summarize_today()
    elif args.period == "week":
        summary = summarizer.summarize_period(start_time, end_time)

    print(f"\n  {Colors.BOLD}{Colors.WHITE}Activity Summary:{Colors.RESET}\n")
    for line in summary.split("\n"):
        print(f"    {line}")
    print()

def cmd_cleanup(args, db):
    """Run retention cleanup."""
    from config import RETENTION_DAYS
    days = args.days or RETENTION_DAYS

    _print_header(f"Retention Cleanup (>{days} days)")

    deleted = db.cleanup_old_captures(retention_days=days)
    print(f"  Deleted {deleted} old captures")


# ─── Main CLI Entry Point ────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ace",
        description="Ambient Context Engine — Privacy-first local context search",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # search
    sp_search = subparsers.add_parser("search", help="Search captured context")
    sp_search.add_argument("query", nargs="+", help="Search query")
    sp_search.add_argument(
        "--mode", choices=["hybrid", "semantic", "keyword"],
        default="hybrid", help="Search mode (default: hybrid)"
    )
    sp_search.add_argument(
        "--limit", type=int, default=10, help="Max results (default: 10)"
    )

    # ask
    sp_ask = subparsers.add_parser("ask", help="Ask a question (RAG)")
    sp_ask.add_argument("question", nargs="+", help="Your question")
    sp_ask.add_argument(
        "--show-sources", action="store_true", default=False,
        help="Show source captures used to answer"
    )

    # summary
    sp_summary = subparsers.add_parser("summary", help="Generate activity summary")
    sp_summary.add_argument(
        "period", nargs="?", default="today", choices=["today", "week"],
        help="Period to summarize (default: today)"
    )

    # recent
    sp_recent = subparsers.add_parser("recent", help="Show recent captures")
    sp_recent.add_argument(
        "count", type=int, nargs="?", default=10,
        help="Number of recent captures to show (default: 10)"
    )

    # status
    subparsers.add_parser("status", help="Show engine status")

    # cleanup
    sp_cleanup = subparsers.add_parser("cleanup", help="Run retention cleanup")
    sp_cleanup.add_argument(
        "--days", type=int, help="Delete captures older than N days"
    )

    return parser


def main():
    """CLI entry point — parses args and dispatches to handlers."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Set up logging
    logging.basicConfig(
        level=logging.WARNING,  # Quiet for CLI usage
        format="%(message)s",
    )

    # Initialize components (lazy — only what's needed)
    from storage.db import DatabaseManager
    db = DatabaseManager()

    try:
        if args.command == "status":
            cmd_status(args, db)

        elif args.command == "recent":
            cmd_recent(args, db)

        elif args.command == "summary":
            cmd_summary(args, db)

        elif args.command == "cleanup":
            cmd_cleanup(args, db)

        elif args.command in ("search", "ask"):
            # These commands need the embedding model + RAG agent
            from processing.embed import EmbeddingGenerator
            from intelligence.rag import RAGAgent

            embedder = EmbeddingGenerator()
            rag = RAGAgent(db, embedder)

            if args.command == "search":
                cmd_search(args, rag)
            else:
                cmd_ask(args, rag)

    finally:
        db.close()


if __name__ == "__main__":
    main()
