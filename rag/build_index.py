import faiss
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from rag.processing_tracker import ProcessingTracker

config = Config()
tracker = ProcessingTracker()

EMBEDDING_FILE = os.path.join(config.EMBED_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(config.EMBED_DIR, "metadata.json")
FAISS_INDEX_FILE = config.FAISS_INDEX

def load_embeddings():
    embeddings = np.load(EMBEDDING_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded {len(embeddings)} embeddings with {len(metadata)} metadata items.")
    return embeddings, metadata

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)  # L2 distance
    index.add(embeddings)
    print(f"FAISS index created with {index.ntotal} vectors of dimension {dim}.")
    return index

def save_index(index, path=FAISS_INDEX_FILE):
    faiss.write_index(index, path)
    print(f"FAISS index saved to {path}.")

def build_index():
    """
    完整的 FAISS index 建立流程
    
    Returns:
        dict: 包含處理結果的字典
    """
    try:
        # Load embeddings
        embeddings, metadata = load_embeddings()
        
        # Create FAISS index
        index = create_faiss_index(embeddings)
        
        # Save index
        index_path = config.FAISS_INDEX
        save_index(index, index_path)
        
        # Mark all files as indexed
        # Get unique source files from metadata
        source_files = set()
        for item in metadata:
            if 'source' in item:
                # Reconstruct the txt file path
                category = item.get('category', '')
                source = item.get('source', '')
                txt_path = os.path.join(config.TXT_DIR, category, source)
                if os.path.exists(txt_path):
                    source_files.add(txt_path)
        
        for txt_file in source_files:
            try:
                tracker.mark_file_completed(txt_file, "indexing")
            except Exception as e:
                print(f"Warning: Could not update tracker for {txt_file}: {str(e)}")
        
        return {
            "success": True,
            "message": "FAISS index built successfully",
            "index_path": index_path,
            "num_vectors": int(index.ntotal),
            "dimension": int(embeddings.shape[1])
        }
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required files not found: {str(e)}. Please run chunking and embedding first.")
    except Exception as e:
        raise Exception(f"Error building index: {str(e)}")

if __name__ == "__main__":
    result = build_index()
    print(json.dumps(result, indent=2))
