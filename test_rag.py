"""
Quick unit/integration test for DocuMind RAG pipeline.
"""
import os
import sys

from parsers import extract_text_from_file
from rag_engine import HybridRAGEngine
from llm_client import LLMClient

def test_pipeline():
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    engine = HybridRAGEngine()
    docs = []

    print("[1] Testing document parsers...")
    for fname in os.listdir(sample_dir):
        fpath = os.path.join(sample_dir, fname)
        if os.path.isfile(fpath):
            doc = extract_text_from_file(fpath, fname)
            docs.append(doc)
            print(f"  ✓ Parsed {fname} ({doc['total_chars']} chars, {doc['file_type']})")

    print(f"\n[2] Indexing {len(docs)} documents into Hybrid RAG Engine...")
    engine.set_corpus(docs)
    print(f"  ✓ Total indexed chunks: {len(engine.all_chunks)}")

    print("\n[3] Testing Hybrid Retrieval (BM25 + TF-IDF RRF)...")
    query = "What are the approved enterprise AI models?"
    results = engine.retrieve(query, top_k=3)
    assert len(results) > 0, "No chunks retrieved!"
    print(f"  ✓ Query: '{query}' -> Top chunk from {results[0]['filename']} (Score: {results[0]['relevance_score']})")

    query2 = "What is the P99 retrieval latency for Project Aurora?"
    results2 = engine.retrieve(query2, top_k=3)
    assert len(results2) > 0, "No chunks retrieved for Aurora!"
    print(f"  ✓ Query: '{query2}' -> Top chunk from {results2[0]['filename']} (Score: {results2[0]['relevance_score']})")
    assert any("< 1.8" in r["text"] or "latency" in r["text"].lower() for r in results2)

    query3 = "Which department exceeded ARR target in Q3?"
    results3 = engine.retrieve(query3, top_k=3)
    assert len(results3) > 0
    print(f"  ✓ Query: '{query3}' -> Top chunk from {results3[0]['filename']} (Score: {results3[0]['relevance_score']})")

    print("\n[4] Testing LLM Extractive Engine (Zero API Key Mode)...")
    llm = LLMClient()
    context = engine.format_context_for_prompt(results)
    ans = llm.generate_response(query, context)
    print(f"  ✓ Generated response ({len(ans['text'])} chars)")
    print("  ✓ Provider:", ans["provider"])

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
