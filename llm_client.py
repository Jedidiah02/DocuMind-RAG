"""
Multi-Provider LLM Client for DocuMind.
Supports Google Gemini, OpenAI, Ollama/Local, and an Offline Extractive Engine.
Uses Python's standard library urllib for zero external dependencies.
"""

import json
import re
import urllib.request
import urllib.error
from typing import List, Dict, Any, Generator, Optional


SYSTEM_RAG_PROMPT = """You are DocuMind, an elite, highly reliable AI document assistant.
Your mission is to assist the user by answering questions, summarizing, and synthesizing information based strictly on the provided document excerpts.

Guidelines:
1. Grounding & Accuracy: Answer primarily using the provided Document Context. Do not invent facts or extrapolate beyond what is documented.
2. Citations: When making a factual claim, cite your source using the exact format: `[Doc: <filename>, Page: <page_num>]`.
3. Clarity & Formatting: Use clean Markdown with headers, bullet points, and bold text for easy reading.
4. If the context does not contain enough information to answer fully, state clearly what is known from the context and what information is missing.
5. Action-Oriented: When asked for summaries, action items, or comparisons, organize them neatly into actionable sections or tables.
"""


class LLMClient:
    def __init__(self):
        self.provider = "gemini"  # "gemini", "openai", "ollama", "offline"
        self.api_key = ""
        self.model = "gemini-1.5-flash"
        self.custom_endpoint = ""
        self.temperature = 0.2

    def configure(self, provider: str = "gemini", api_key: str = "", model: str = "", custom_endpoint: str = "", temperature: float = 0.2):
        self.provider = provider.lower()
        self.api_key = api_key.strip()
        self.model = model.strip() or ("gemini-1.5-flash" if self.provider == "gemini" else "gpt-4o-mini")
        self.custom_endpoint = custom_endpoint.strip()
        self.temperature = temperature

    def generate_response(self, query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generates a complete response using the configured provider or offline fallback."""
        
        # If no API key provided and not using local Ollama, use intelligent offline extractor
        if not self.api_key and self.provider not in ["ollama", "offline"]:
            return self._offline_extractive_answer(query, context)

        if self.provider == "gemini":
            return self._call_gemini(query, context, history)
        elif self.provider in ["openai", "ollama"]:
            return self._call_openai_compatible(query, context, history)
        else:
            return self._offline_extractive_answer(query, context)

    def _call_gemini(self, query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Call Google Gemini REST API directly with urllib."""
        if not self.api_key:
            return self._offline_extractive_answer(query, context)

        model_name = self.model or "gemini-1.5-flash"
        # Support gemini-2.5-flash, gemini-1.5-flash, etc.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"

        # Construct Gemini contents array
        contents = []
        
        # System instructions in Gemini format
        system_instruction = {
            "parts": [{"text": SYSTEM_RAG_PROMPT}]
        }

        # Add history if provided
        if history:
            for turn in history[-6:]:  # last 3 pairs
                role = "user" if turn.get("role") == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": turn.get("content", "")}]
                })

        user_content = f"### DOCUMENT CONTEXT:\n{context}\n\n### USER QUERY:\n{query}"
        contents.append({
            "role": "user",
            "parts": [{"text": user_content}]
        })

        payload = {
            "contents": contents,
            "system_instruction": system_instruction,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 2048
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    answer_text = "".join([p.get("text", "") for p in parts])
                    return {
                        "text": answer_text,
                        "provider": "Google Gemini",
                        "model": model_name,
                        "success": True
                    }
                else:
                    return {
                        "text": "Gemini returned no response content. The input may have been flagged by safety filters.",
                        "provider": "Google Gemini",
                        "model": model_name,
                        "success": False
                    }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            try:
                err_json = json.loads(error_body)
                err_msg = err_json.get("error", {}).get("message", error_body)
            except Exception:
                err_msg = error_body
            return {
                "text": f"**Gemini API Error ({e.code})**: {err_msg}\n\n*Falling back to local extractive analysis below:*\n\n" + self._offline_extractive_answer(query, context)["text"],
                "provider": "Google Gemini (Fallback)",
                "model": model_name,
                "success": False
            }
        except Exception as e:
            return {
                "text": f"**Connection Error**: {str(e)}\n\n*Falling back to local extractive analysis below:*\n\n" + self._offline_extractive_answer(query, context)["text"],
                "provider": "Offline Fallback",
                "model": "local",
                "success": False
            }

    def _call_openai_compatible(self, query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Calls OpenAI or any OpenAI-compatible API (Ollama, Groq, OpenRouter)."""
        if self.provider == "ollama":
            url = self.custom_endpoint or "http://localhost:11434/v1/chat/completions"
            auth_header = "Bearer ollama"
        else:
            url = self.custom_endpoint or "https://api.openai.com/v1/chat/completions"
            auth_header = f"Bearer {self.api_key}"

        messages = [{"role": "system", "content": SYSTEM_RAG_PROMPT}]
        if history:
            for turn in history[-6:]:
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})

        messages.append({
            "role": "user",
            "content": f"### DOCUMENT CONTEXT:\n{context}\n\n### USER QUERY:\n{query}"
        })

        payload = {
            "model": self.model or ("llama3" if self.provider == "ollama" else "gpt-4o-mini"),
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 2048
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": auth_header
                }
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    return {
                        "text": content,
                        "provider": self.provider.title(),
                        "model": self.model,
                        "success": True
                    }
                return {
                    "text": "Received empty response from LLM endpoint.",
                    "provider": self.provider.title(),
                    "model": self.model,
                    "success": False
                }
        except Exception as e:
            return {
                "text": f"**API Error ({self.provider})**: {str(e)}\n\n*Falling back to local extractive analysis below:*\n\n" + self._offline_extractive_answer(query, context)["text"],
                "provider": "Offline Fallback",
                "model": "local",
                "success": False
            }

    def _offline_extractive_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        Intelligent offline extractive synthesizer.
        Extracts key sentences and structured lines matching the query and semantic synonyms,
        formats date timelines when asked temporal questions, and provides citations.
        """
        if not context or context.strip() == "No relevant context found in documents.":
            return {
                "text": "I reviewed the available documents, but could not find information relevant to your question. Please verify your uploaded documents or try a different search keyword.",
                "provider": "DocuMind Extractive Engine (Zero API Key)",
                "model": "offline-extractive",
                "success": True
            }

        q_lower = query.lower()
        q_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', q_lower))
        
        # Domain Synonyms expansion for extraction
        if any(w in q_lower for w in ["study", "education", "degree", "college", "university", "academics"]):
            q_words.update(["education", "degree", "college", "university", "mca", "bachelor", "master", "b.sc", "bsc", "graduat"])
        if any(w in q_lower for w in ["work", "job", "career", "experience", "intern"]):
            q_words.update(["experience", "role", "company", "responsibilities", "intern", "qa"])
        if any(w in q_lower for w in ["finish", "complete", "when", "year", "date"]):
            q_words.update(["2024", "2022", "2023", "2025", "jul", "nov", "jan", "june", "year", "duration"])

        is_temporal = any(w in q_lower for w in ["when", "year", "date", "period", "duration", "timeline", "finish", "graduat"])

        # Break context into chunks
        blocks = context.split("---")
        extracted_points = []
        direct_timeline = []

        for block in blocks:
            raw_lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
            if not raw_lines:
                continue
            header = raw_lines[0] if raw_lines[0].startswith("[Source") else "Relevant Section"
            body_lines = raw_lines[1:] if raw_lines[0].startswith("[Source") else raw_lines

            matching_lines = []
            for line in body_lines:
                line_lower = line.lower()
                matches = sum(1 for w in q_words if w in line_lower)
                
                # Check for temporal matches (e.g. Nov 2022 – Jul 2024)
                has_date = bool(re.search(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b|\b\d{4}\s*[-–—]\s*\d{4}\b|\b(19|20)\d{2}\b', line_lower))
                
                if is_temporal and has_date and matches > 0:
                    matches += 3
                    direct_timeline.append(line)
                    
                if matches > 0:
                    matching_lines.append((matches, line))

            matching_lines.sort(key=lambda x: x[0], reverse=True)
            top_lines = [l for _, l in matching_lines[:4]]
            
            if top_lines:
                extracted_points.append({
                    "header": header,
                    "lines": top_lines
                })

        output_lines = [
            "> ℹ️ *Tip: You're running in Local Extractive Mode. To enable full conversational generative synthesis, click ⚙️ Settings and add your Google Gemini or OpenAI API Key.*",
            "",
            f"### Key Findings for: *\"{query}\"*",
            ""
        ]

        # If a temporal question was asked and we found specific timeline lines, show a direct summary card
        if direct_timeline:
            output_lines.append("#### 📅 Relevant Timeline & Completion Dates:")
            seen = set()
            for item in direct_timeline:
                if item not in seen:
                    seen.add(item)
                    output_lines.append(f"- **{item}**")
            output_lines.append("")

        if extracted_points:
            for item in extracted_points:
                output_lines.append(f"#### 📄 {item['header']}")
                for l in item['lines']:
                    output_lines.append(f"- {l}")
                output_lines.append("")
        else:
            # Fallback to direct excerpt summary
            output_lines.append("Here are the most relevant excerpts retrieved from your documents:\n")
            for b in blocks[:3]:
                output_lines.append(b.strip() + "\n")

        output_lines.append("\n**Suggested Next Steps:**")
        output_lines.append("- Ask for another section, project details, or certifications.")
        output_lines.append("- Use the quick action buttons on the left to generate an instant Executive Summary or Action Items.")

        return {
            "text": "\n".join(output_lines),
            "provider": "DocuMind Extractive Engine (Zero API Key)",
            "model": "offline-extractive",
            "success": True
        }
