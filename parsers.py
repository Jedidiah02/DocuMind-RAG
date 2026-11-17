"""
Universal Document Parser for DocuMind RAG Assistant.
Handles PDF, DOCX, TXT, MD, CSV, JSON, and source code files.
Works with 100% standard library (optional enhanced support if pypdf is installed).
"""

import os
import re
import csv
import json
import zlib
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple

# Check for optional pypdf
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


def extract_text_from_file(file_path: str, filename: str) -> Dict[str, Any]:
    """
    Extracts text and metadata from a given file.
    Returns:
        {
            "filename": filename,
            "file_type": ext,
            "total_chars": int,
            "pages": [{"page_num": int, "text": str}],
            "full_text": str,
            "metadata": dict
        }
    """
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    
    if ext in ["txt", "md", "markdown", "rst", "log", "py", "js", "ts", "html", "css", "sql", "yaml", "yml", "xml"]:
        return _parse_text(file_path, filename, ext)
    elif ext == "csv" or ext == "tsv":
        return _parse_csv(file_path, filename, ext)
    elif ext == "json":
        return _parse_json(file_path, filename, ext)
    elif ext == "docx":
        return _parse_docx(file_path, filename)
    elif ext == "pdf":
        return _parse_pdf(file_path, filename)
    else:
        # Fallback to general text attempt
        try:
            return _parse_text(file_path, filename, ext)
        except Exception as e:
            raise ValueError(f"Unsupported or unreadable file format: .{ext} ({str(e)})")


def _read_with_fallback(file_path: str) -> str:
    """Read a text file with multiple encoding fallbacks."""
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _parse_text(file_path: str, filename: str, ext: str) -> Dict[str, Any]:
    text = _read_with_fallback(file_path)
    return {
        "filename": filename,
        "file_type": ext,
        "total_chars": len(text),
        "pages": [{"page_num": 1, "text": text}],
        "full_text": text,
        "metadata": {"format": "plain_text", "lines": len(text.splitlines())}
    }


def _parse_csv(file_path: str, filename: str, ext: str) -> Dict[str, Any]:
    raw_text = _read_with_fallback(file_path)
    delimiter = "\t" if ext == "tsv" else ","
    
    # Try reading as CSV and format as readable markdown table + textual summaries
    lines = raw_text.splitlines()
    if not lines:
        return {"filename": filename, "file_type": ext, "total_chars": 0, "pages": [{"page_num": 1, "text": ""}], "full_text": "", "metadata": {}}
    
    reader = csv.reader(lines, delimiter=delimiter)
    rows = [row for row in reader if row]
    
    if not rows:
        text = raw_text
    else:
        headers = rows[0]
        formatted_blocks = []
        formatted_blocks.append(f"# Data Table: {filename}\n")
        formatted_blocks.append(f"Headers: {', '.join(headers)}\nTotal Records: {len(rows)-1}\n")
        
        # Build Markdown representation for chunking
        # Add markdown table for small/medium files
        if len(rows) <= 100:
            md_table = "| " + " | ".join(headers) + " |\n"
            md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for row in rows[1:100]:
                padded_row = row + [""] * (len(headers) - len(row))
                md_table += "| " + " | ".join([c.replace("|", "/") for c in padded_row[:len(headers)]]) + " |\n"
            formatted_blocks.append(md_table)
        
        # Also format each record as key-value pairs for precise semantic retrieval
        formatted_blocks.append("\n## Records Detail:\n")
        for i, row in enumerate(rows[1:], 1):
            record_str = f"Record #{i}: " + ", ".join([f"{headers[j]}: {row[j]}" for j in range(min(len(headers), len(row)))])
            formatted_blocks.append(record_str)
            
        text = "\n".join(formatted_blocks)
        
    return {
        "filename": filename,
        "file_type": ext,
        "total_chars": len(text),
        "pages": [{"page_num": 1, "text": text}],
        "full_text": text,
        "metadata": {"rows": len(rows), "columns": len(headers) if rows else 0}
    }


def _parse_json(file_path: str, filename: str, ext: str) -> Dict[str, Any]:
    raw_text = _read_with_fallback(file_path)
    try:
        data = json.loads(raw_text)
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        text = f"# JSON Document: {filename}\n\n```json\n{formatted}\n```"
    except Exception:
        text = raw_text

    return {
        "filename": filename,
        "file_type": ext,
        "total_chars": len(text),
        "pages": [{"page_num": 1, "text": text}],
        "full_text": text,
        "metadata": {"format": "json"}
    }


def _parse_docx(file_path: str, filename: str) -> Dict[str, Any]:
    """Pure-Python DOCX parser using zipfile and xml parsing (zero dependencies)."""
    text_pieces = []
    
    with zipfile.ZipFile(file_path, 'r') as docx_zip:
        xml_content = docx_zip.read('word/document.xml')
        root = ET.fromstring(xml_content)
        
        # Namespace map for Word XML
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        for p in root.iter(f"{{{ns['w']}}}p"):
            p_text = "".join(node.text for node in p.iter(f"{{{ns['w']}}}t") if node.text)
            if p_text.strip():
                text_pieces.append(p_text.strip())
                
        # Also look for tables
        for tbl in root.iter(f"{{{ns['w']}}}tbl"):
            tbl_rows = []
            for tr in tbl.iter(f"{{{ns['w']}}}tr"):
                row_cells = []
                for tc in tr.iter(f"{{{ns['w']}}}tc"):
                    cell_text = "".join(node.text for node in tc.iter(f"{{{ns['w']}}}t") if node.text)
                    row_cells.append(cell_text.strip())
                if any(row_cells):
                    tbl_rows.append(" | ".join(row_cells))
            if tbl_rows:
                text_pieces.append("\n[Table]\n" + "\n".join(tbl_rows) + "\n[/Table]")

    full_text = "\n\n".join(text_pieces)
    return {
        "filename": filename,
        "file_type": "docx",
        "total_chars": len(full_text),
        "pages": [{"page_num": 1, "text": full_text}],
        "full_text": full_text,
        "metadata": {"paragraphs": len(text_pieces)}
    }


def _parse_pdf(file_path: str, filename: str) -> Dict[str, Any]:
    """PDF parser with pypdf and fallback handling."""
    # Attempt dynamic import of pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        pages = []
        full_text_list = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages.append({"page_num": idx + 1, "text": page_text})
            if page_text.strip():
                full_text_list.append(f"--- Page {idx + 1} ---\n{page_text}")
        
        full_text = "\n\n".join(full_text_list).strip()
        
        # If pypdf extracted text successfully
        if full_text:
            return {
                "filename": filename,
                "file_type": "pdf",
                "total_chars": len(full_text),
                "pages": pages,
                "full_text": full_text,
                "metadata": {"page_count": len(pages), "engine": "pypdf"}
            }
        else:
            notice = (
                f"Notice: PDF '{filename}' contains 0 extractable text characters (it may be a scanned document, "
                f"image-only PDF, or have text converted into vector curves). Please upload a digital PDF with selectable text, "
                f"or a DOCX/TXT format."
            )
            return {
                "filename": filename,
                "file_type": "pdf",
                "total_chars": len(notice),
                "pages": [{"page_num": 1, "text": notice}],
                "full_text": notice,
                "metadata": {"page_count": len(pages), "engine": "pypdf", "scanned_or_vector": True}
            }
    except Exception as e:
        pass

    # Pure Python basic PDF text stream extractor
    pages = []
    full_text_list = []
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        # Extract text objects using regex on uncompressed or zlib decompressed stream blocks
        stream_matches = re.findall(b"stream[\r\n]+(.*?)[\r\n]+endstream", pdf_bytes, re.DOTALL)
        extracted_strings = []
        
        for stream in stream_matches:
            decompressed = None
            try:
                decompressed = zlib.decompress(stream)
            except Exception:
                decompressed = stream # Maybe not flate-encoded

            if decompressed:
                # Search for TJ and Tj text arrays
                # e.g. (Hello World) Tj
                # or [(Hello) -10 (World)] TJ
                tjs = re.findall(rb"\((.*?)\)\s*Tj", decompressed)
                for tj in tjs:
                    try:
                        extracted_strings.append(tj.decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
                
                # Arrays: [(Some) 20 (Text)] TJ
                tjs_arr = re.findall(rb"\[(.*?)\]\s*TJ", decompressed)
                for arr in tjs_arr:
                    parts = re.findall(rb"\((.*?)\)", arr)
                    combined = "".join([p.decode("utf-8", errors="ignore") for p in parts])
                    if combined.strip():
                        extracted_strings.append(combined)

        extracted_text = " ".join(extracted_strings).strip()
        # Clean up repeated whitespace
        extracted_text = re.sub(r"[ \t]+", " ", extracted_text)
        extracted_text = re.sub(r"\n\s*\n+", "\n\n", extracted_text)
        
        if not extracted_text:
            extracted_text = f"Notice: PDF '{filename}' could not be decoded with the built-in parser. You can install pypdf via 'pip install pypdf' for full PDF support, or upload text/markdown/docx versions."

        pages.append({"page_num": 1, "text": extracted_text})
        return {
            "filename": filename,
            "file_type": "pdf",
            "total_chars": len(extracted_text),
            "pages": pages,
            "full_text": extracted_text,
            "metadata": {"page_count": 1, "engine": "built-in stream parser"}
        }
    except Exception as err:
        return {
            "filename": filename,
            "file_type": "pdf",
            "total_chars": 0,
            "pages": [{"page_num": 1, "text": f"Error parsing PDF: {str(err)}"}],
            "full_text": f"Error parsing PDF: {str(err)}",
            "metadata": {"error": str(err)}
        }
