import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from config import Config

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_chunks(directory):
    """Load all chunk JSON files and return a flat list of texts."""
    texts = []
    metadata = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    text = item.get("text", "").strip()
                    if not text:
                        continue                
                    texts.append(text)
                    metadata.append(item)
    return texts, metadata

def embed_texts(texts, model_name=config.EMBEDDING_MODEL):
    """Embed a list of texts with SentenceTransformer."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device)

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    print("Embedding completed!")
    return embeddings

def save_embedding(embeddings, metadata, output_dir=config.EMBED_DIR):
    os.makedirs(output_dir, exist_ok=True)
    # Save embeddings
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2) 
    print(f"Saved embeddings and metadata to {output_dir}")


if __name__ == "__main__":
    # 1) Load chunks
    chunk_dir = config.CHUNK_DIR  
    texts, metadata = load_chunks(chunk_dir)
    print(f"Loaded {len(texts)} chunks.")
    # 2) Run embeddings
    embeddings = embed_texts(texts)
    # 3) Example output
    print("Example Text:", texts[0][:200], "...")
    print("Embedding vector shape:", embeddings.shape)
    print("First 5 dims of embedding:", embeddings[0][:5])
    # 4) Saves embedding
    save_embedding(embeddings, metadata)
