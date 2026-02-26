import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from rag.processing_tracker import ProcessingTracker

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
tracker = ProcessingTracker()

def load_chunks(directory, only_unprocessed=True):
    """
    Load chunk JSON files and return a flat list of texts.
    
    Args:
        directory: 目錄路徑
        only_unprocessed: 是否只載入未處理過的檔案
    
    Returns:
        texts: 文字列表
        metadata: 元資料列表
        source_files: 來源檔案列表（用於追蹤）
    """
    texts = []
    metadata = []
    source_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".json"):
                file_path = os.path.join(root, file)
                
                # 找出對應的原始 .txt 檔案
                txt_file = file_path.replace(config.CHUNK_DIR, config.TXT_DIR).replace(".json", ".txt")
                
                # 檢查是否需要跳過
                if only_unprocessed and tracker.is_file_processed(txt_file, stage="embedding"):
                    print(f"Skipped (already embedded): {txt_file}")
                    continue
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    chunk_start_idx = len(texts)
                    for item in data:
                        text = item.get("text", "").strip()
                        if not text:
                            continue                
                        texts.append(text)
                        metadata.append(item)
                    
                    chunk_end_idx = len(texts) - 1
                    source_files.append({
                        "txt_file": txt_file,
                        "json_file": file_path,
                        "chunk_range": (chunk_start_idx, chunk_end_idx),
                        "chunk_count": chunk_end_idx - chunk_start_idx + 1
                    })
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                    
    return texts, metadata, source_files

def embed_texts(texts, model_name=config.EMBEDDING_MODEL):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device)

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    print("Embedding completed!")
    return embeddings

def save_embedding(embeddings, metadata, output_dir=config.EMBED_DIR, mode="append"):
    """
    儲存 embeddings 和 metadata
    
    Args:
        embeddings: 新的 embeddings
        metadata: 新的 metadata
        output_dir: 輸出目錄
        mode: 'append' 增量追加 或 'overwrite' 覆寫
    """
    os.makedirs(output_dir, exist_ok=True)
    
    embedding_path = os.path.join(output_dir, "embeddings.npy")
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    if mode == "append" and os.path.exists(embedding_path) and os.path.exists(metadata_path):
        # 載入現有資料
        existing_embeddings = np.load(embedding_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
        
        # 合併新舊資料
        combined_embeddings = np.vstack([existing_embeddings, embeddings])
        combined_metadata = existing_metadata + metadata
        
        # 儲存合併後的資料
        np.save(embedding_path, combined_embeddings)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(combined_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Appended {len(embeddings)} new embeddings (total: {len(combined_embeddings)})")
    else:
        # 覆寫模式或首次建立
        np.save(embedding_path, embeddings)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(embeddings)} embeddings to {output_dir}")


if __name__ == "__main__":
    # 1) Load chunks (只載入未處理的)
    chunk_dir = config.CHUNK_DIR  
    texts, metadata, source_files = load_chunks(chunk_dir, only_unprocessed=True)
    
    if len(texts) == 0:
        print("All files have been processed. No new embeddings needed.")
        print(f"\nStatistics:")
        stats = tracker.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    else:
        print(f"Loaded {len(texts)} chunks from {len(source_files)} files.")
        
        # 標記開始處理
        for source in source_files:
            tracker.mark_file_processing(source["txt_file"], "embedding")
        
        try:
            # 2) Run embeddings
            embeddings = embed_texts(texts)
            
            # 3) Example output
            print("Example Text:", texts[0][:200], "...")
            print("Embedding vector shape:", embeddings.shape)
            print("First 5 dims of embedding:", embeddings[0][:5])
            
            # 4) Save embeddings (append mode)
            save_embedding(embeddings, metadata, mode="append")
            
            # 5) 標記所有檔案處理完成
            current_idx = 0
            existing_count = 0
            
            # 計算現有的 embedding 數量
            embedding_path = os.path.join(config.EMBED_DIR, "embeddings.npy")
            if os.path.exists(embedding_path):
                existing_embeddings = np.load(embedding_path)
                existing_count = len(existing_embeddings) - len(embeddings)
            
            for source in source_files:
                start_idx = existing_count + source["chunk_range"][0]
                end_idx = existing_count + source["chunk_range"][1]
                
                tracker.mark_file_completed(
                    source["txt_file"],
                    "embedding",
                    embedding_range=(start_idx, end_idx),
                    embedding_count=source["chunk_count"]
                )
                print(f"Marked {source['txt_file']} as completed (embeddings {start_idx}-{end_idx})")
                
        except Exception as e:
            # 標記失敗
            for source in source_files:
                tracker.mark_file_failed(source["txt_file"], "embedding", str(e))
            print(f"Embedding failed: {str(e)}")
            raise