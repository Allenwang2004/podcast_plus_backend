#!/usr/bin/env python3
"""
Persistent RAG retrieval worker service
Keeps the embedding model loaded in memory and processes retrieval tasks from a queue
"""
import sys
import json
import os
import time
from pathlib import Path

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RetrievalWorkerService:
    def __init__(self):
        """Initialize worker with embedding model"""
        print("[Retrieval Service] Initializing...", file=sys.stderr)
        
        # Pre-load the retrieval system to avoid repeated model loading
        from rag.retrieval import retrieve
        self.retrieve_fn = retrieve
        
        # Do a dummy retrieval to ensure models are loaded
        try:
            _ = self.retrieve_fn("test", top_n=1)
            print("[Retrieval Service] Models loaded and ready", file=sys.stderr)
        except Exception as e:
            print(f"[Retrieval Service] Warning during initialization: {e}", file=sys.stderr)
        
        self.is_running = True
        
    def retrieve_context(self, query: str, top_n: int = 3):
        """Retrieve context from FAISS index"""
        try:
            print(f"[Retrieval Service] Processing query: {query[:50]}...", file=sys.stderr)
            
            # Retrieve relevant chunks
            results = self.retrieve_fn(query, top_n=top_n)
            
            if results:
                # Print detailed information about retrieved chunks
                print(f"[Retrieval Service] Retrieved {len(results)} chunks:", file=sys.stderr)
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
                
                return {
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
                print("[Retrieval Service] No relevant chunks found", file=sys.stderr)
                return {
                    "success": True,
                    "context": "",
                    "num_chunks": 0,
                    "chunks_info": []
                }
            
        except Exception as e:
            print(f"[Retrieval Service] Error: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_queue(self):
        """Process retrieval tasks from queue"""
        queue_dir = Path(__file__).parent / "queue"
        queue_dir.mkdir(exist_ok=True)
        
        print("[Retrieval Service] Watching queue directory...", file=sys.stderr)
        
        while self.is_running:
            try:
                # Check for retrieval task files
                task_files = sorted(queue_dir.glob("retrieve_task_*.json"))
                
                if task_files:
                    task_file = task_files[0]
                    
                    try:
                        # Read task
                        with open(task_file, 'r', encoding='utf-8') as f:
                            task = json.load(f)
                        
                        task_id = task['task_id']
                        
                        # Process task
                        result = self.retrieve_context(
                            query=task['query'],
                            top_n=task.get('top_n', 3)
                        )
                        
                        # Write result
                        result_file = queue_dir / f"retrieve_result_{task_id}.json"
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f)
                        
                        # Remove task file
                        task_file.unlink()
                        
                    except Exception as e:
                        print(f"[Retrieval Service] Failed to process {task_file}: {e}", file=sys.stderr)
                        task_file.unlink()
                else:
                    # No tasks, sleep briefly
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n[Retrieval Service] Shutting down...", file=sys.stderr)
                self.is_running = False
                break
            except Exception as e:
                print(f"[Retrieval Service] Queue processing error: {e}", file=sys.stderr)
                time.sleep(1)

    def run(self):
        """Start the retrieval service"""
        print("[Retrieval Service] Service started", file=sys.stderr)
        self.process_queue()
        print("[Retrieval Service] Service stopped", file=sys.stderr)

if __name__ == "__main__":
    service = RetrievalWorkerService()
    service.run()
