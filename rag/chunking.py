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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

processed_count = 0
skipped_count = 0

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".txt"):
            file_path = os.path.join(root, file)
            
            # 檢查是否已處理過
            if tracker.is_file_processed(file_path, stage="chunking"):
                print(f"Skipped (already processed): {file_path}")
                skipped_count += 1
                continue
            
            try:
                # 標記開始處理
                tracker.mark_file_processing(file_path, "chunking")
                
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = text_splitter.split_text(text)

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
                    
                rel_dir = os.path.relpath(root, input_dir)
                out_dir_full = os.path.join(output_dir, rel_dir)
                os.makedirs(out_dir_full, exist_ok=True)

                out_path = os.path.join(out_dir_full, file.replace(".txt", ".json"))
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # 標記處理完成
                tracker.mark_file_completed(
                    file_path, 
                    "chunking",
                    chunk_count=len(data),
                    output_path=out_path
                )
                
                print(f"Saved {len(data)} chunks -> {out_path}")
                processed_count += 1
                
            except Exception as e:
                # 標記處理失敗
                tracker.mark_file_failed(file_path, "chunking", str(e))
                print(f"Failed to process {file_path}: {str(e)}")

print(f"\n{'='*60}")
print(f"Chunking Summary:")
print(f"   Processed: {processed_count} files")
print(f"   Skipped: {skipped_count} files")
print(f"{'='*60}")