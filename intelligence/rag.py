"""
Intelligence Layer — RAG (Retrieval-Augmented Generation) Agent

Answers natural-language questions about your activity history by:
1. Embedding the question
2. Running hybrid search (semantic + keyword) over the captures database
3. Constructing a context-rich prompt
4. Generating an answer via Ollama (local LLM)

Zero cloud dependency — everything runs on localhost.
"""

import json
import logging
import time
from typing import Optional

import requests
import numpy as np

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_CONTEXT_MAX_CHARS, SEARCH_TOP_K,
)
from storage.db import DatabaseManager
from processing.embed import EmbeddingGenerator

logger = logging.getLogger(__name__)

# ─── Prompt Templates ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a personal productivity assistant with access to the user's captured screen text and audio transcripts. Your job is to answer questions about what the user has been working on, reading, or discussing.

Rules:
- Answer based ONLY on the provided context. Do not make things up.
- Cite timestamps and window/app names when referencing specific activities.
- If the context doesn't contain enough information to answer, say so honestly.
- Be concise and actionable in your responses.
- Format timestamps in a human-readable way (e.g., "today at 2:30 PM" or "yesterday at 10 AM")."""

CONTEXT_TEMPLATE = """Here are the user's recent captured activities, ordered by relevance:

{context_entries}

---
User's question: {question}

Answer based on the captured context above:"""


class RAGAgent:
    """Local RAG agent that queries the captures database and generates
    answers using a local LLM via Ollama."""

    def __init__(self, db: DatabaseManager, embedder: EmbeddingGenerator):
        self.db = db
        self.embedder = embedder
        self._verify_ollama()

    def _verify_ollama(self):
        """Check that Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]

            if OLLAMA_MODEL not in model_names:
                # Try matching without tag
                base_names = [m["name"].split(":")[0] for m in models]
                if OLLAMA_MODEL.split(":")[0] not in base_names:
                    logger.warning(
                        f"Model '{OLLAMA_MODEL}' not found in Ollama. "
                        f"Available: {model_names}. "
                        f"Run: ollama pull {OLLAMA_MODEL}"
                    )
                    return

            logger.info(f"Ollama verified: model '{OLLAMA_MODEL}' available")

        except requests.ConnectionError:
            logger.warning(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            logger.warning(f"Ollama verification issue: {e}")

    def _format_context(self, results: list[dict]) -> str:
        """Format search results into a context string for the LLM prompt."""
        entries = []
        total_chars = 0

        for i, r in enumerate(results, 1):
            source_icon = "🖥️" if r["source"] == "screen" else "🎤"
            window_info = ""
            if r.get("window_title"):
                window_info = f" | Window: {r['window_title']}"

            text = r["text_content"]

            # Truncate individual entries if needed
            if total_chars + len(text) > LLM_CONTEXT_MAX_CHARS:
                remaining = LLM_CONTEXT_MAX_CHARS - total_chars
                if remaining < 100:
                    break
                text = text[:remaining] + "..."

            entry = (
                f"[{i}] {source_icon} {r['timestamp']}{window_info}\n"
                f"    {text}"
            )
            entries.append(entry)
            total_chars += len(text)

        return "\n\n".join(entries)

    def query(self, question: str, top_k: int = SEARCH_TOP_K) -> dict:
        """
        Answer a question about the user's activity history.

        Returns a dict with:
            - answer: The LLM-generated answer
            - sources: The retrieved context entries
            - search_time_ms: How long retrieval took
            - generate_time_ms: How long LLM generation took
        """
        total_start = time.time()

        # Step 1: Embed the question
        query_embedding = self.embedder.generate(question)

        # Step 2: Hybrid search
        search_start = time.time()
        results = self.db.search_hybrid(question, query_embedding, limit=top_k)
        search_time = (time.time() - search_start) * 1000

        if not results:
            return {
                "answer": "I don't have any captured context to answer that question. "
                          "Make sure the capture pipeline is running.",
                "sources": [],
                "search_time_ms": search_time,
                "generate_time_ms": 0,
            }

        # Step 3: Build the prompt
        context_str = self._format_context(results)
        prompt = CONTEXT_TEMPLATE.format(
            context_entries=context_str,
            question=question,
        )

        # Step 4: Generate answer via Ollama
        gen_start = time.time()
        answer = self._call_ollama(prompt)
        gen_time = (time.time() - gen_start) * 1000

        total_time = (time.time() - total_start) * 1000
        logger.info(
            f"RAG query completed in {total_time:.0f}ms "
            f"(search={search_time:.0f}ms, generate={gen_time:.0f}ms)"
        )

        return {
            "answer": answer,
            "sources": results,
            "search_time_ms": search_time,
            "generate_time_ms": gen_time,
        }

    def search_only(
        self, query_text: str, mode: str = "hybrid", top_k: int = SEARCH_TOP_K
    ) -> list[dict]:
        """
        Search without LLM generation. Useful for the `search` CLI command.

        Args:
            query_text: Search query
            mode: 'hybrid', 'semantic', or 'keyword'
            top_k: Number of results
        """
        query_embedding = self.embedder.generate(query_text)

        if mode == "semantic":
            return self.db.search_semantic(query_embedding, limit=top_k)
        elif mode == "keyword":
            return self.db.search_keyword(query_text, limit=top_k)
        else:
            return self.db.search_hybrid(query_text, query_embedding, limit=top_k)

    def _call_ollama(self, prompt: str) -> str:
        """Call the Ollama API to generate a response."""
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 512,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()

            data = resp.json()
            return data.get("response", "").strip()

        except requests.ConnectionError:
            return (
                "⚠️  Cannot connect to Ollama. Make sure it's running: "
                "`ollama serve`"
            )
        except requests.Timeout:
            return "⚠️  Ollama request timed out. The model may be loading."
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"⚠️  Error generating answer: {e}"


# ─── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("Testing RAG agent...")
    print("  (Requires Ollama running with llama3.2:3b pulled)")

    # Initialize components
    db = DatabaseManager()
    embedder = EmbeddingGenerator()
    rag = RAGAgent(db, embedder)

    # Insert some test data
    test_entries = [
        ("screen", "Working on the database schema for the ambient context engine",
         "schema.sql - VS Code", "Code.exe"),
        ("screen", "Reviewing Python documentation for threading module",
         "threading — Python docs - Chrome", "chrome.exe"),
        ("audio", "Let's discuss the project timeline and milestones for next week",
         None, None),
    ]

    for source, text, window, process in test_entries:
        emb = embedder.generate(text)
        db.insert_capture(
            source=source,
            text_content=text,
            embedding=emb,
            window_title=window,
            process_name=process,
        )

    print("  Inserted test data")

    # Test search
    results = rag.search_only("what was I coding?")
    print(f"\n  Search 'what was I coding?': {len(results)} results")
    for r in results:
        print(f"    [{r['source']}] {r['text_content'][:60]}...")

    # Test RAG query
    print("\n  Asking RAG: 'What was I working on today?'")
    response = rag.query("What was I working on today?")
    print(f"\n  Answer: {response['answer']}")
    print(f"  Search time: {response['search_time_ms']:.0f}ms")
    print(f"  Generate time: {response['generate_time_ms']:.0f}ms")

    db.close()
    print("\nRAG test complete!")
