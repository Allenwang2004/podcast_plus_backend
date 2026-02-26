from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import sys
import os
import faiss
import numpy as np
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import Config
from app.src.schema.rag_schema import (
    BuildIndexRequest, 
    BuildIndexResponse,
    ChunkAndEmbedRequest,
    ChunkAndEmbedResponse,
    IndexStatusResponse
)
from rag.chunking import text_splitter
from rag.embedding import load_chunks, embed_texts, save_embedding
from rag.build_index import load_embeddings, create_faiss_index, save_index

router = APIRouter()
config = Config()

@router.post("/build-index", response_model=BuildIndexResponse)
async def build_faiss_index(request: BuildIndexRequest = None):
    """
    Build FAISS index from existing embeddings.
    
    This endpoint loads pre-computed embeddings and creates a FAISS index.
    The embeddings must already exist in the embeddings directory.
    """
    try:
        # Load embeddings
        embeddings, metadata = load_embeddings()
        
        # Create FAISS index
        index = create_faiss_index(embeddings)
        
        # Save index
        index_path = config.FAISS_INDEX
        save_index(index, index_path)
        
        return BuildIndexResponse(
            success=True,
            message="FAISS index built successfully",
            index_path=index_path,
            num_vectors=index.ntotal,
            dimension=embeddings.shape[1]
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Required files not found: {str(e)}. Please run chunking and embedding first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building index: {str(e)}")


@router.post("/chunk-and-embed", response_model=ChunkAndEmbedResponse)
async def chunk_and_embed_texts(request: ChunkAndEmbedRequest):
    """
    Chunk text files and create embeddings.
    
    This endpoint processes text files from the specified directory,
    splits them into chunks, and generates embeddings for each chunk.
    """
    try:
        # Get parameters from request or use defaults
        text_dir = request.text_directory or config.TXT_DIR
        chunk_dir = config.CHUNK_DIR
        embedding_model = request.embedding_model or config.EMBEDDING_MODEL
        
        # Check if text directory exists
        if not os.path.exists(text_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Text directory not found: {text_dir}"
            )
        
        # Step 1: Chunking (using the logic from chunking.py)
        os.makedirs(chunk_dir, exist_ok=True)
        total_chunks = 0
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import re
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        for root, dirs, files in os.walk(text_dir):
            for file in files:
                if file.lower().endswith(".txt"):
                    file_path = os.path.join(root, file)
                    
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
                    
                    total_chunks += len(data)
        
        # Step 2: Load chunks and create embeddings
        texts, metadata = load_chunks(chunk_dir)
        
        # Step 3: Generate embeddings
        embeddings = embed_texts(texts, model_name=embedding_model)
        
        # Step 4: Save embeddings
        save_embedding(embeddings, metadata, output_dir=config.EMBED_DIR)
        
        return ChunkAndEmbedResponse(
            success=True,
            message="Chunking and embedding completed successfully",
            num_chunks=total_chunks,
            num_embeddings=len(embeddings),
            embedding_shape=list(embeddings.shape)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chunking and embedding: {str(e)}")


@router.post("/full-pipeline", response_model=dict)
async def full_pipeline(request: ChunkAndEmbedRequest):
    """
    Execute the full RAG pipeline: chunking -> embedding -> index building.
    
    This endpoint runs all steps sequentially to create a complete FAISS index
    from text files.
    """
    try:
        # Step 1: Chunk and embed
        chunk_embed_response = await chunk_and_embed_texts(request)
        
        # Step 2: Build index
        build_response = await build_faiss_index()
        
        return {
            "success": True,
            "message": "Full RAG pipeline completed successfully",
            "chunking_embedding": {
                "num_chunks": chunk_embed_response.num_chunks,
                "num_embeddings": chunk_embed_response.num_embeddings,
                "embedding_shape": chunk_embed_response.embedding_shape
            },
            "index_building": {
                "index_path": build_response.index_path,
                "num_vectors": build_response.num_vectors,
                "dimension": build_response.dimension
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in full pipeline: {str(e)}")


@router.get("/status", response_model=IndexStatusResponse)
async def get_index_status():
    """
    Get the status of the FAISS index.
    
    Returns information about whether the index exists and its properties.
    """
    try:
        index_path = config.FAISS_INDEX
        
        if not os.path.exists(index_path):
            return IndexStatusResponse(
                exists=False,
                path=index_path,
                message="FAISS index does not exist"
            )
        
        # Load index to get info
        index = faiss.read_index(index_path)
        
        return IndexStatusResponse(
            exists=True,
            path=index_path,
            num_vectors=index.ntotal,
            dimension=index.d,
            message="FAISS index exists and is ready"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking index status: {str(e)}")


@router.delete("/index")
async def delete_index():
    """
    Delete the existing FAISS index.
    """
    try:
        index_path = config.FAISS_INDEX
        
        if not os.path.exists(index_path):
            raise HTTPException(status_code=404, detail="Index file does not exist")
        
        os.remove(index_path)
        
        return {
            "success": True,
            "message": f"Index deleted successfully from {index_path}"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting index: {str(e)}")
