# 🧠 Privacy-First Local Ambient Context Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/SQLite-sqlite--vec-green.svg)](https://github.com/asgregorio/sqlite-vec)
[![LLM](https://img.shields.io/badge/Local%20LLM-Ollama%20%2F%20llama3.2%3A3b-orange.svg)](https://ollama.ai/)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20On--Device-brightgreen.svg)]()

> A 100% local, privacy-first background engine that continuously records screen and system audio context, performs real-time OCR and speech-to-text (STT), indexes sliding-window vector embeddings into SQLite via `sqlite-vec`, and serves sub-second contextual AI search and proactive nudges without cloud telemetry.

*Inspired by [mediar-ai/screenpipe](https://github.com/mediar-ai/screenpipe) and [Logseq](https://github.com/logseq/logseq).*

---

## 🌟 Key Features

- **📷 Continuous Screen Capture & Deduplication**: Frame-differencing algorithm captures active window frames every 3 seconds, skipping redundant frames.
- **🎙️ Ambient Audio Transcription**: Captures microphone/system audio in 30-second sliding chunks and transcribes using quantized Whisper STT.
- **🔒 Automated PII Redaction**: On-the-fly regex-based masking of emails, phone numbers, and credit cards before indexing to protect personal data.
- **⚡ Hybrid Vector & Keyword Search**: Combines `sqlite-vec` KNN vector search (`all-MiniLM-L6-v2`) with SQLite FTS5 keyword indexing using Reciprocal Rank Fusion (RRF).
- **🕸️ Context Knowledge Graph**: Tracks temporal and application relationships between screen actions and voice events.
- **💡 Proactive Nudge Agent**: Analyzes recent user focus sessions to surface helpful break reminders, task summaries, or context insights via Ollama (`llama3.2:3b`).
- **🎨 Modern Web Dashboard**: Sleek, dark-mode workspace (`http://localhost:5000`) featuring session timelines, statistics, and a persistent RAG chat search bar.

---

## 🏗️ System Architecture

```
┌─────────────────┐       ┌─────────────────┐
│ Screen Capture  │       │  Audio Capture  │
│ (mss / pywin32) │       │  (sounddevice)  │
└────────┬────────┘       └────────┬────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│  Tesseract OCR  │       │   Whisper STT   │
│ (Frame Diffing) │       │ (Audio Chunks)  │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └──────────┬──────────────┘
                    │
                    ▼
         ┌───────────────────┐
         │   PII Redactor    │
         └──────────┬────────┘
                    │
                    ▼
         ┌───────────────────┐
         │ SentenceTransformer│
         │ (all-MiniLM-L6-v2)│
         └──────────┬────────┘
                    │
                    ▼
         ┌───────────────────┐
         │ SQLite + vec0/FTS5│ (WAL Mode)
         └──────────┬────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌─────────────────┐   ┌──────────────────┐
│  Local RAG Chat │   │ Proactive Nudges │
│ (Ollama 3.2:3b) │   │ (Intelligence)   │
└────────┬────────┘   └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │ Web UI (localhost:5000)│
       └────────────────────────┘
```

---

## 🛠️ Prerequisites

Before getting started, make sure you have the following installed on your system:

1. **Python 3.10+** (64-bit)
2. **Tesseract OCR**
   - **Windows**: Download installer from [UB-Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki) and install to `C:\Program Files\Tesseract-OCR\tesseract.exe`.
   - **Linux/macOS**: `sudo apt install tesseract-ocr` or `brew install tesseract`.
3. **Ollama** (for RAG and Nudge Agent)
   - Download from [ollama.ai](https://ollama.ai/).
   - Pull the default model:
     ```bash
     ollama pull llama3.2:3b
     ```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ambient-context-engine.git
cd ambient-context-engine
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration (Optional)
Copy `.env.example` to `.env` if custom paths or variables are needed:
```bash
cp .env.example .env
```

### 5. Run the Application
Start the pipeline and Web UI server:
```bash
python main.py
```

> **Note for Windows Security Policies**: If your machine blocks compiled audio DLLs (AppLocker / Windows Defender policy), run in screen-only mode:
> ```bash
> python main.py --no-audio
> ```

---

## 🖥️ Web UI Dashboard

Once `main.py` is running, open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

### Dashboard Highlights:
- **Left Navigation**: Switch between Chat view and Activity Timeline.
- **Centered Workspace**: Displays "Day Recap" and AI-driven "Proactive Insights".
- **Bottom Chat Input**: Query your local context (e.g. *"What python functions did I work on today?"* or *"What did I read in documentation?"*).

---

## ⚙️ CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--no-audio` | Disable audio capture (Screen-only mode) | `False` |
| `--no-screen` | Disable screen capture (Audio-only mode) | `False` |
| `--log-level` | Set logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

---

## 📁 Repository Structure

```
ambient-context-engine/
├── capture/                    # Input Capture Layer
│   ├── screen_capture.py       # MSS multi-monitor capture & frame diffing
│   └── audio_capture.py        # PyAudio / sounddevice audio buffer stream
├── processing/                 # Processing & Transformation Layer
│   ├── ocr.py                  # Tesseract OCR processor & deduplication
│   ├── transcribe.py           # Speech-To-Text transcription engine
│   ├── embed.py                # SentenceTransformers vector generation
│   └── redact.py               # Regex-based PII masking
├── storage/                    # Database Layer
│   ├── schema.sql              # FTS5 + vec0 + sessions + graph table definitions
│   └── db.py                   # SQLite connection manager & RRF hybrid search
├── intelligence/               # AI & RAG Layer
│   ├── rag.py                  # Local RAG query agent (Ollama interface)
│   └── nudge_agent.py          # Proactive insight & session evaluator
├── interface/                  # Web Dashboard Layer
│   ├── app.py                  # Flask REST API server
│   ├── templates/index.html    # Single-page dashboard application
│   └── static/                 # Stylesheet (style.css) & Frontend logic (script.js)
├── config.py                   # Centralized application parameters
├── main.py                     # Pipeline orchestrator & thread manager
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment variables
└── README.md                   # Project documentation
```

---

## 🤝 Collaboration Guidelines

1. **Branching Strategy**: Create feature branches from `main` (e.g., `feature/custom-ocr`, `fix/audio-buffer`).
2. **Local Testing**: Always run `python main.py` locally to verify that database tables create smoothly and vector extensions load without issues.
3. **Commit Messages**: Keep commit messages concise and descriptive.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
