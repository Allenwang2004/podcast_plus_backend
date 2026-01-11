import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn import pipeline
from config import Config
import os
import pandas as pd

# File paths
config = Config()
FAISS_INDEX_FILE = config.FAISS_INDEX
METADATA_FILE = os.path.join(config.EMBED_DIR, "metadata.json")
EMBEDDING_MODEL = config.EMBEDDING_MODEL
RERANK_MODEL = config.RERANK_MODEL
TOP_K = config.TOP_K
TOP_N = config.TOP_N
ALPHA = config.ALPHA

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_FILE)

# Load metadata
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embedding model
encoder = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

def retrieve(query, top_k=TOP_K, top_n=TOP_N, alpha=ALPHA):
    """Retrieve top-k chunks for a query. Import this for context."""
    query_vec = encoder.encode([query], convert_to_numpy=True)
    # Normalize for cosine similarity
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    distances, indices = index.search(query_vec, top_k)
    candidates = [metadata[i] for i in indices[0]]
    faiss_scores = distances[0]

    cross_inputs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(cross_inputs)

    hybrid_scores = alpha * faiss_scores + (1 - alpha) * rerank_scores

    ranked = sorted(zip(hybrid_scores, candidates), key=lambda x: x[0], reverse=True)

    return [c for s, c in ranked[:top_n]]


def process_queries_with_summary(input_csv, output_csv):
    queries = pd.read_csv(input_csv)['question'].tolist()

    all_results = []

    for query in queries:
        results = retrieve(query)
        retrieved_texts = [res['text'] for res in results]
        context = "\n".join(retrieved_texts[:3])  # 取前3個 chunk 作為 context

        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nPlease summarize or answer based on the context above."

        all_results.append({
            'query': query,
            'retrieved_text': context,
        })

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    test_query = pd.read_csv('evaluation/queries_rag.csv')['question'].tolist()
    for q in test_query:
        results = retrieve(q)
        df = pd.DataFrame(results[0], index=[0])