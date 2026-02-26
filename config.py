import torch
import os

class Config:
    # 取得項目根目錄的絕對路徑
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    VECTOR_DB_PATH = "embeddings/faiss_index.bin"
    TOP_K = 20
    TOP_N = 5
    ALPHA = 0.75

    #RAG config
    PDF_DIR = [
        os.path.join(ROOT_DIR, "documents/Computer/"),
        os.path.join(ROOT_DIR, "documents/Physics/"),
        os.path.join(ROOT_DIR, "documents/Probability/")
    ]
    TXT_DIR = os.path.join(ROOT_DIR, "db/text")
    CHUNK_DIR = os.path.join(ROOT_DIR, "db/chunks")
    EMBED_DIR = os.path.join(ROOT_DIR, "db/embeddings")
    FAISS_INDEX = os.path.join(ROOT_DIR, "db/faiss_index.index")
    
    # Fine-tuning config
    OUTPUT_DIR = "./llama-dialogue-finetuned"
    LORA_RANK = 32
    LORA_ALPHA = 64
    NUM_EPOCHS = 3
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    MAX_SEQ_LENGTH = 512
    GRAD_ACCUMULATION = 8
    WEIGHT_DECAY = 0.1
    MAX_GRAD_NORM = 1.0
    
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Prompt Template
    # Topic -> Context (RAG) -> Generation
    PROMPT_TEMPLATE = """You are a helpful assistant capable of generating coherent dialogues based on a topic and context.

### Topic:
{topic}

### Context (Reference):
{context}

### Dialogue:
"""