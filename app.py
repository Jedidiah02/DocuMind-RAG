"""
DocuMind RAG Assistant Server.
Built with Python 3 Standard Library (Zero External Dependencies Required).
Serves REST API, Hybrid RAG Engine, and Modern Interactive Web Application.
"""

import os
import sys
import json
import base64
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

from parsers import extract_text_from_file
from rag_engine import HybridRAGEngine
from llm_client import LLMClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SAMPLE_DOCS_DIR = os.path.join(BASE_DIR, "sample_docs")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Global instances
rag_engine = HybridRAGEngine(chunk_size=700, chunk_overlap=120)
llm_client = LLMClient()
raw_documents_store = {}  # filename -> parsed_doc_dict


def load_initial_samples():
    """Load sample documents and any existing uploaded documents into RAG engine."""
    # 1. Load sample docs
    if os.path.exists(SAMPLE_DOCS_DIR):
        for fname in os.listdir(SAMPLE_DOCS_DIR):
            fpath = os.path.join(SAMPLE_DOCS_DIR, fname)
            if os.path.isfile(fpath) and fname not in raw_documents_store:
                try:
                    parsed = extract_text_from_file(fpath, fname)
                    raw_documents_store[fname] = parsed
                except Exception as e:
                    print(f"Error loading sample {fname}: {e}")

    # 2. Load uploaded docs
    if os.path.exists(UPLOADS_DIR):
        for fname in os.listdir(UPLOADS_DIR):
            fpath = os.path.join(UPLOADS_DIR, fname)
            if os.path.isfile(fpath) and fname not in raw_documents_store:
                try:
                    parsed = extract_text_from_file(fpath, fname)
                    raw_documents_store[fname] = parsed
                except Exception as e:
                    print(f"Error loading uploaded file {fname}: {e}")
                    
    rag_engine.set_corpus(list(raw_documents_store.values()))
    print(f"Loaded {len(raw_documents_store)} total documents with {len(rag_engine.all_chunks)} total chunks.")


class DocuMindRequestHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]}\n")

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400):
        self._send_json({"error": message, "success": False}, status=status)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # API Routes
        if path == "/api/documents":
            docs_summary = []
            for fname, d in raw_documents_store.items():
                docs_summary.append({
                    "filename": fname,
                    "file_type": d.get("file_type", ""),
                    "total_chars": d.get("total_chars", 0),
                    "chunk_count": len([c for c in rag_engine.all_chunks if c["filename"] == fname]),
                    "page_count": len(d.get("pages", [])),
                    "metadata": d.get("metadata", {})
                })
            self._send_json({
                "documents": docs_summary,
                "total_chunks": len(rag_engine.all_chunks)
            })
            return

        elif path == "/api/documents/content":
            fname = query_params.get("filename", [""])[0]
            if fname in raw_documents_store:
                doc = raw_documents_store[fname]
                chunks = [c for c in rag_engine.all_chunks if c["filename"] == fname]
                self._send_json({
                    "filename": fname,
                    "full_text": doc.get("full_text", "")[:15000],  # preview limit
                    "truncated": len(doc.get("full_text", "")) > 15000,
                    "total_chars": doc.get("total_chars", 0),
                    "chunks": chunks
                })
            else:
                self._send_error_json("Document not found", status=404)
            return

        elif path == "/api/settings":
            self._send_json({
                "provider": llm_client.provider,
                "model": llm_client.model,
                "has_api_key": bool(llm_client.api_key),
                "api_key_masked": ("..." + llm_client.api_key[-4:]) if len(llm_client.api_key) > 6 else ("Set" if llm_client.api_key else "Not Set"),
                "custom_endpoint": llm_client.custom_endpoint,
                "temperature": llm_client.temperature,
                "chunk_size": rag_engine.chunker.chunk_size,
                "chunk_overlap": rag_engine.chunker.chunk_overlap
            })
            return

        # Serve static assets
        self._serve_static(path)

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b""
        
        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            body = {}

        if path == "/api/documents/upload":
            filename = body.get("filename", "").strip()
            content_base64 = body.get("content_base64", "")
            content_text = body.get("content_text", "")

            if not filename:
                self._send_error_json("Missing filename")
                return

            # Sanitize filename
            safe_filename = os.path.basename(filename)
            save_path = os.path.join(UPLOADS_DIR, safe_filename)

            try:
                if content_base64:
                    file_bytes = base64.b64decode(content_base64)
                    with open(save_path, "wb") as f:
                        f.write(file_bytes)
                elif content_text:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(content_text)
                else:
                    self._send_error_json("No file content received")
                    return

                # Parse document
                parsed_doc = extract_text_from_file(save_path, safe_filename)
                raw_documents_store[safe_filename] = parsed_doc
                # Refresh index
                rag_engine.set_corpus(list(raw_documents_store.values()))

                chunk_count = len([c for c in rag_engine.all_chunks if c["filename"] == safe_filename])
                self._send_json({
                    "success": True,
                    "filename": safe_filename,
                    "chunk_count": chunk_count,
                    "total_chars": parsed_doc.get("total_chars", 0),
                    "total_corpus_chunks": len(rag_engine.all_chunks)
                })
            except Exception as e:
                self._send_error_json(f"Failed to process file: {str(e)}", status=500)
            return

        elif path == "/api/documents/load-samples":
            load_initial_samples()
            self._send_json({
                "success": True,
                "document_count": len(raw_documents_store),
                "total_chunks": len(rag_engine.all_chunks)
            })
            return

        elif path == "/api/documents/clear":
            raw_documents_store.clear()
            rag_engine.clear()
            self._send_json({"success": True, "message": "All documents removed."})
            return

        elif path == "/api/chat":
            query = body.get("query", "").strip()
            filter_doc = body.get("filter_filename") or None
            history = body.get("history", [])
            top_k = int(body.get("top_k", 5))

            if not query:
                self._send_error_json("Empty query")
                return

            if not rag_engine.all_chunks:
                self._send_json({
                    "answer": "No documents are currently indexed. Please upload documents or click **'Load Sample Documents'** to start asking questions!",
                    "citations": [],
                    "provider": "DocuMind",
                    "model": "system"
                })
                return

            # Hybrid Retrieval
            retrieved_chunks = rag_engine.retrieve(query, top_k=top_k, filter_filename=filter_doc)
            context = rag_engine.format_context_for_prompt(retrieved_chunks)

            # LLM Generation
            result = llm_client.generate_response(query=query, context=context, history=history)

            # Format Citations
            citations = []
            for c in retrieved_chunks:
                citations.append({
                    "chunk_id": c["chunk_id"],
                    "filename": c["filename"],
                    "page_num": c["page_num"],
                    "relevance_score": c["relevance_score"],
                    "snippet": c["text"][:280] + ("..." if len(c["text"]) > 280 else "")
                })

            self._send_json({
                "answer": result.get("text", ""),
                "citations": citations,
                "provider": result.get("provider", "DocuMind"),
                "model": result.get("model", ""),
                "success": result.get("success", True)
            })
            return

        elif path == "/api/tools/quick-action":
            action = body.get("action", "")  # "summary", "actions", "faq", "compare"
            filter_doc = body.get("filter_filename") or None

            prompts = {
                "summary": "Provide a comprehensive Executive Summary of the document(s). Highlight main goals, key findings, and strategic takeaways in bullet points.",
                "actions": "Extract all Action Items, Next Steps, Deliverables, and Deadlines mentioned in the document(s). Present them in a structured table or checklist with Assignee/Role if available.",
                "faq": "Formulate the Top 5 most critical Questions and Answers (FAQ) someone reading these document(s) would ask. Provide concise, clear answers.",
                "risks": "Analyze and extract all Risks, Constraints, Bottlenecks, and Compliance requirements outlined in the document(s)."
            }

            query = prompts.get(action, "Summarize the key information.")
            
            # Retrieve more context for document tools
            retrieved_chunks = rag_engine.retrieve(query, top_k=7, filter_filename=filter_doc)
            context = rag_engine.format_context_for_prompt(retrieved_chunks)
            result = llm_client.generate_response(query=query, context=context)

            citations = [{
                "chunk_id": c["chunk_id"],
                "filename": c["filename"],
                "page_num": c["page_num"],
                "relevance_score": c["relevance_score"],
                "snippet": c["text"][:250] + "..."
            } for c in retrieved_chunks]

            self._send_json({
                "action": action,
                "answer": result.get("text", ""),
                "citations": citations,
                "provider": result.get("provider", "DocuMind")
            })
            return

        elif path == "/api/settings":
            provider = body.get("provider", "gemini")
            api_key = body.get("api_key", "")
            model = body.get("model", "")
            custom_endpoint = body.get("custom_endpoint", "")
            temperature = float(body.get("temperature", 0.2))
            chunk_size = int(body.get("chunk_size", 700))
            chunk_overlap = int(body.get("chunk_overlap", 120))

            llm_client.configure(
                provider=provider,
                api_key=api_key,
                model=model,
                custom_endpoint=custom_endpoint,
                temperature=temperature
            )

            # Update chunker parameters if changed
            if chunk_size != rag_engine.chunker.chunk_size or chunk_overlap != rag_engine.chunker.chunk_overlap:
                rag_engine.chunker.chunk_size = chunk_size
                rag_engine.chunker.chunk_overlap = chunk_overlap
                rag_engine.set_corpus(list(raw_documents_store.values()))

            self._send_json({"success": True, "message": "Settings updated successfully."})
            return

        self._send_error_json("Unknown API endpoint", status=404)

    def do_DELETE(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path.startswith("/api/documents/"):
            filename = urllib.parse.unquote(path.replace("/api/documents/", "")).strip()
            if filename in raw_documents_store:
                del raw_documents_store[filename]
                rag_engine.set_corpus(list(raw_documents_store.values()))
                self._send_json({"success": True, "removed": filename})
            else:
                self._send_error_json("File not found in store", status=404)
            return

        self._send_error_json("Endpoint not found", status=404)

    def _serve_static(self, path: str):
        if path in ["/", ""]:
            path = "/index.html"
            
        clean_path = path.lstrip("/")
        full_path = os.path.join(STATIC_DIR, clean_path)

        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            self.send_error(404, f"File not found: {path}")
            return

        ext = os.path.splitext(full_path)[1].lower()
        mime_types = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".ico": "image/x-icon"
        }
        content_type = mime_types.get(ext, "application/octet-stream")

        with open(full_path, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(port: int = 8080):
    load_initial_samples()
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, DocuMindRequestHandler)
    print(f"\n=======================================================")
    print(f"🚀 DocuMind RAG Assistant is running!")
    print(f"👉 Open in your browser: http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down DocuMind server...")
        httpd.server_close()


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
