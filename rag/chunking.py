import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from rag.processing_tracker import ProcessingTracker

config = Config()
input_dir = config.TXT_DIR
output_dir = config.CHUNK_DIR 
os.makedirs(output_dir, exist_ok=True)

# 初始化處理追蹤器
tracker = ProcessingTracker()

def chunk_texts(text_dir=None, chunk_size=500, chunk_overlap=50):
    """
    將文本檔案切分為 chunks
    
    Args:
        text_dir: 文本目錄路徑
        chunk_size: chunk 大小
        chunk_overlap: chunk 重疊大小
    
    Returns:
        dict: 包含處理結果的字典
    """
    text_dir = text_dir or input_dir
    chunk_dir = output_dir
    
    # Check if text directory exists
    if not os.path.exists(text_dir):
        raise FileNotFoundError(f"Text directory not found: {text_dir}")
    
    os.makedirs(chunk_dir, exist_ok=True)
    total_chunks = 0
    processed_count = 0
    skipped_count = 0
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for root, dirs, files in os.walk(text_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                file_path = os.path.join(root, file)
                
                # Check if already processed
                if tracker.is_file_processed(file_path, stage="chunking"):
                    print(f"Skipping (already processed): {file_path}")
                    skipped_count += 1
                    continue
                
                try:
                    # Mark file as processing
                    tracker.mark_file_processing(file_path, "chunking")
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    chunks = splitter.split_text(text)
                    
                    data = []
                    chunk_id = 0
                    last_page = 1
                    
                    for idx, chunk in enumerate(chunks):
                        page_match = re.search(r'\[Page (\d+)\]', chunk)
                        if page_match:
                            last_page = int(page_match.group(1))
                        page_num = last_page
                        chunk = re.sub(r'\[Page \d+\]', '', chunk).strip()
                        if not chunk:
                            continue
                        category = os.path.basename(root)
                        source_path = os.path.relpath(file_path, root).replace("\\", "/")
                        
                        data.append({
                            "source": source_path,
                            "category": category,
                            "chunk_index": chunk_id,
                            "page": page_num,
                            "text": chunk
                        })
                        chunk_id += 1
                    
                    rel_dir = os.path.relpath(root, text_dir)
                    out_dir_full = os.path.join(chunk_dir, rel_dir)
                    os.makedirs(out_dir_full, exist_ok=True)
                    
                    out_path = os.path.join(out_dir_full, file.replace(".txt", ".json"))
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    # Mark file as completed
                    tracker.mark_file_completed(
                        file_path,
                        "chunking",
                        chunk_count=len(data),
                        output_path=out_path
                    )
                    
                    total_chunks += len(data)
                    processed_count += 1
                    print(f"Processed: {file_path} -> {len(data)} chunks")
                    
                except Exception as e:
                    # Mark file as failed
                    tracker.mark_file_failed(file_path, "chunking", str(e))
                    print(f"Failed to process {file_path}: {str(e)}")
                    raise
    
    return {
        "success": True,
        "message": f"Chunking completed: {processed_count} processed, {skipped_count} skipped",
        "num_chunks": total_chunks,
        "processed_files": processed_count,
        "skipped_files": skipped_count
    }

### Testing script
if __name__ == "__main__":
    result = chunk_texts(
        text_dir=config.TXT_DIR,
        chunk_size=500,
        chunk_overlap=50
    )
    print(json.dumps(result, indent=2))