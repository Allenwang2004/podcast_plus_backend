import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import sys
import re
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


def embed_chunks(chunk_dir=None, embedding_model=None):
    """
    為 chunks 生成 embeddings
    
    Args:
        chunk_dir: chunk 目錄路徑
        embedding_model: 使用的 embedding 模型名稱
    
    Returns:
        dict: 包含處理結果的字典
    """
    chunk_dir = chunk_dir or config.CHUNK_DIR
    embedding_model = embedding_model or config.EMBEDDING_MODEL
    
    tracker = ProcessingTracker()
    
    # Load chunks
    texts, metadata, source_files = load_chunks(chunk_dir, only_unprocessed=True)
    
    if len(texts) == 0:
        print("No new texts to embed")
        return {
            "success": True,
            "message": "No new chunks to embed",
            "num_embeddings": 0,
            "embedding_shape": [0, 0]
        }
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = embed_texts(texts, model_name=embedding_model)
    
    # Save embeddings
    save_embedding(embeddings, metadata, output_dir=config.EMBED_DIR, mode="append")
    
    # Mark embedding as completed for all source files
    for source_file_info in source_files:
        try:
            tracker.mark_file_completed(
                source_file_info["txt_file"],
                "embedding",
                chunk_count=source_file_info["chunk_count"]
            )
        except Exception as e:
            print(f"Warning: Could not update tracker for {source_file_info['txt_file']}: {str(e)}")
    
    return {
        "success": True,
        "message": "Embedding completed successfully",
        "num_embeddings": len(embeddings),
        "embedding_shape": list(embeddings.shape)
    }

### Testing script
if __name__ == "__main__":
    result = embed_chunks(
        chunk_dir=config.CHUNK_DIR,
        embedding_model=config.EMBEDDING_MODEL
    )
    print(json.dumps(result, indent=2))