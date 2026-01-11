import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from config import Config

config = Config()
input_dir = config.TXT_DIR
output_dir = config.CHUNK_DIR 
os.makedirs(output_dir, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".txt"):
            file_path = os.path.join(root, file)

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
                
            print(f"Saved {len(data)} chunks -> {out_path}")