#!/usr/bin/env python3
"""
Standalone RAG retrieval worker
Run retrieval in a separate process to avoid tokenizer fork warnings
"""
import sys
import json
import os

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def retrieve_context(query: str, top_n: int = 3):
    """Retrieve context from FAISS index"""
    try:
        print(f"[Worker] Retrieving context for: {query[:50]}...", file=sys.stderr)
        
        from rag.retrieval import retrieve
        
        # Retrieve relevant chunks
        results = retrieve(query, top_n=top_n)
        
        if results:
            # Print detailed information about retrieved chunks
            print(f"[Worker] Retrieved {len(results)} chunks:", file=sys.stderr)
            for idx, chunk in enumerate(results, 1):
                source = chunk.get('source', 'unknown')
                category = chunk.get('category', 'unknown')
                page = chunk.get('page', 'N/A')
                text_preview = chunk.get('text', '')[:100].replace('\n', ' ')
                print(f"  [{idx}] {category}/{source} (Page {page})", file=sys.stderr)
                print(f"      Preview: {text_preview}...", file=sys.stderr)
            
            # Combine retrieved chunks into context
            retrieved_texts = [chunk['text'] for chunk in results]
            context = "\n\n".join(retrieved_texts)
            
            result = {
                "success": True,
                "context": context,
                "num_chunks": len(results),
                "chunks_info": [
                    {
                        "source": chunk.get('source', 'unknown'),
                        "category": chunk.get('category', 'unknown'),
                        "page": chunk.get('page', 'N/A')
                    }
                    for chunk in results
                ]
            }
        else:
            print("[Worker] No relevant chunks found", file=sys.stderr)
            result = {
                "success": True,
                "context": "",
                "num_chunks": 0,
                "chunks_info": []
            }
        
        print(json.dumps(result))
        return 0
        
    except Exception as e:
        print(f"[Worker] Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(result))
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: retrieve_worker.py <query> [top_n]", file=sys.stderr)
        sys.exit(1)
    
    query = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    sys.exit(retrieve_context(query, top_n))
