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

if __name__ == "__main__":
    try:
        print("🔨 Building FAISS index...")
        embeddings, metadata = load_embeddings()
        index = create_faiss_index(embeddings)
        save_index(index)
        
        # 標記完整重建
        tracker.mark_full_rebuild()
        
        # 將所有已嵌入的檔案標記為已索引
        print("\nUpdating processing log...")
        for item in metadata:
            # 從 metadata 取得原始檔案路徑
            source = item.get("source", "")
            category = item.get("category", "")
            
            # 重建完整的 txt 檔案路徑
            txt_file = os.path.join(config.TXT_DIR, category, source)
            
            # 檢查檔案是否存在且已完成 embedding
            if os.path.exists(txt_file):
                file_info = tracker.get_file_info(txt_file)
                if file_info and file_info.get("embedding_status") == "completed":
                    # 標記為已索引（不重複標記已完成的）
                    if file_info.get("index_status") != "completed":
                        tracker.mark_file_completed(txt_file, "indexing")
        
        print("\nFAISS index build completed!")
        
        # 顯示統計資訊
        stats = tracker.get_statistics()
        print(f"\nProcessing Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"Error building index: {str(e)}")
        raise
