import faiss
import numpy as np
import json
import os
from config import Config

config = Config()

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
    embeddings, metadata = load_embeddings()
    index = create_faiss_index(embeddings)
    save_index(index)
