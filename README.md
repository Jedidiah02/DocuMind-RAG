# DocuMind — Universal RAG Chatbot & Document Assistant

**DocuMind** is a production-grade, zero-friction Retrieval-Augmented Generation (RAG) assistant designed for everyone. It runs immediately on any Windows, macOS, or Linux machine with **Python 3 alone (zero external pip packages required to launch)**, and provides a sleek, modern glassmorphic web interface.

---

## ✨ Key Capabilities

1. **Universal Document Ingestion**:
   - Drag & drop or upload files: `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.json`, `.py`, `.js`, etc.
   - Built-in pure-Python parsers for Word docs (`.docx`), plain text, Markdown, tabular CSVs, and PDFs.
   - Preserves page numbers, table structures, and metadata.

2. **State-of-the-Art Hybrid RAG Engine**:
   - **BM25 Lexical Ranking**: High-speed keyword matching with term frequency, inverse document frequency (IDF), and document length normalization.
   - **TF-IDF & Semantic Similarity**: Computes local normalized vectors with cosine similarity.
   - **Reciprocal Rank Fusion (RRF)**: Merges sparse lexical and dense semantic results using $\frac{1}{60 + \text{rank}}$ for maximum retrieval precision.
   - **Grounding & Clickable Citations**: Every claim is cited with `[Doc: filename, Page: X]`. Clicking any citation badge opens an inspector showing the exact source excerpt and relevance score.

3. **Multi-Model Support with Zero-Setup Fallback**:
   - **Google Gemini API** (Free tier via `aistudio.google.com` - `gemini-1.5-flash`, `gemini-2.5-flash`, `gemini-1.5-pro`)
   - **OpenAI API** (`gpt-4o`, `gpt-4o-mini`)
   - **Local Ollama** (100% private offline LLMs like `llama3`, `mistral`, `deepseek`)
   - **Built-in Offline Extractive Engine**: Works immediately with **0 API keys**! Accurately retrieves and highlights exact answering sentences and paragraphs directly from documents.

4. **1-Click Document Assistant Tools**:
   - 📑 **Executive Summary**: Instant bulleted brief of main goals and findings.
   - 🎯 **Action Items & To-Dos**: Automatically extracts tasks, owners, and deadlines.
   - ❓ **Auto FAQ**: Generates the top 5 questions and answers from the documents.
   - ⚠️ **Risk & Compliance Audit**: Discovers bottlenecks, risks, and compliance criteria.

---

## 🚀 Quick Start Guide

### 1. Launch the Server
In PowerShell or Terminal:
```powershell
cd C:\Users\dimpl\.gemini\antigravity\scratch\documind-rag
python app.py
```

### 2. Open in Your Browser
Navigate to:
```
http://localhost:8080
```

### 3. Immediate Testing
- Click **"Load 3 Sample Docs"** in the sidebar to instantly load:
  - `enterprise_ai_policy.md` (AI Governance & compliance standards)
  - `project_aurora_specs.txt` (Distributed storage architecture & SLAs)
  - `q3_financial_metrics.csv` (Revenue, churn, ARR, and department targets)
- Click any of the suggested prompt cards or ask your own question!

---

## 🛠️ Architecture Overview

```
documind-rag/
├── app.py                # Threaded HTTP Server & REST API endpoints
├── parsers.py            # Universal document parser (DOCX, PDF, CSV, TXT, JSON)
├── rag_engine.py         # Chunking, BM25, TF-IDF, Vector Cosine, and RRF Fusion
├── llm_client.py         # Multi-provider LLM connector (Gemini, OpenAI, Ollama, Offline)
├── sample_docs/          # Preloaded sample enterprise documents
│   ├── enterprise_ai_policy.md
│   ├── project_aurora_specs.txt
│   └── q3_financial_metrics.csv
├── static/
│   ├── index.html        # Responsive SPA UI (Tailwind CSS, Lucide icons, Modals)
│   ├── app.js            # Client state, drag-drop uploads, RAG chat, citations
│   └── styles.css        # Typography, markdown styles, citation pills
└── requirements.txt      # Optional packages for enhanced PDF parsing
```

---

## 📦 Optional Enhancements
To enable enhanced PDF parsing for complex compressed or scanned PDFs:
```powershell
pip install pypdf
```
*(The app automatically detects and uses `pypdf` when available, falling back gracefully to the built-in parser when not.)*
