from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
import sys
import os
import faiss
import numpy as np
import json
import shutil
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import Config
from app.src.schema.rag_schema import (
    BuildIndexRequest, 
    BuildIndexResponse,
    ChunkAndEmbedRequest,
    ChunkAndEmbedResponse,
    IndexStatusResponse,
    UploadPDFResponse
)
from rag.chunking import text_splitter
from rag.embedding import load_chunks, embed_texts, save_embedding
from rag.build_index import load_embeddings, create_faiss_index, save_index
from rag.text_extraction import extract_text, extract_text_ocr, save_text
from rag.processing_tracker import ProcessingTracker

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
        # Initialize tracker
        tracker = ProcessingTracker()
        
        # Load embeddings
        embeddings, metadata = load_embeddings()
        
        # Create FAISS index
        index = create_faiss_index(embeddings)
        
        # Save index
        index_path = config.FAISS_INDEX
        save_index(index, index_path)
        
        # Mark all files as indexed
        # Get unique source files from metadata
        source_files = set()
        for item in metadata:
            if 'source' in item:
                # Reconstruct the txt file path
                category = item.get('category', '')
                source = item.get('source', '')
                txt_path = os.path.join(config.TXT_DIR, category, source)
                if os.path.exists(txt_path):
                    source_files.add(txt_path)
        
        for txt_file in source_files:
            try:
                tracker.mark_file_completed(txt_file, "indexing")
            except Exception as e:
                print(f"Warning: Could not update tracker for {txt_file}: {str(e)}")
        
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
        
        # Initialize processing tracker
        tracker = ProcessingTracker()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        for root, dirs, files in os.walk(text_dir):
            for file in files:
                if file.lower().endswith(".txt"):
                    file_path = os.path.join(root, file)
                    
                    # Check if already processed
                    if tracker.is_file_processed(file_path, stage="chunking"):
                        print(f"Skipping (already processed): {file_path}")
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
                        print(f"Processed: {file_path} -> {len(data)} chunks")
                        
                    except Exception as e:
                        # Mark file as failed
                        tracker.mark_file_failed(file_path, "chunking", str(e))
                        print(f"Failed to process {file_path}: {str(e)}")
                        raise
        
        # Step 2: Load chunks and create embeddings
        texts, metadata, source_files = load_chunks(chunk_dir, only_unprocessed=True)
        
        if len(texts) == 0:
            print("No new texts to embed")
            return ChunkAndEmbedResponse(
                success=True,
                message="No new chunks to process",
                num_chunks=total_chunks,
                num_embeddings=0,
                embedding_shape=[0, 0]
            )
        
        # Step 3: Generate embeddings
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = embed_texts(texts, model_name=embedding_model)
        
        # Step 4: Save embeddings
        save_embedding(embeddings, metadata, output_dir=config.EMBED_DIR, mode="append")
        
        # Step 5: Mark embedding as completed for all source files
        for source_file_info in source_files:
            try:
                tracker.mark_file_completed(
                    source_file_info["txt_file"],
                    "embedding",
                    chunk_count=source_file_info["chunk_count"]
                )
            except Exception as e:
                print(f"Warning: Could not update tracker for {source_file_info['txt_file']}: {str(e)}")
        
        return ChunkAndEmbedResponse(
            success=True,
            message="Chunking and embedding completed successfully",
            num_chunks=total_chunks,
            num_embeddings=len(embeddings),
            embedding_shape=list(embeddings.shape)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chunking and embedding: {str(e)}")


@router.post("/upload-pdf", response_model=UploadPDFResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    auto_process: bool = Form(False)
):
    """
    Upload a PDF file and optionally process it automatically.
    
    - **file**: PDF file to upload
    - **auto_process**: If True, automatically extract text, chunk, embed, and build index
    
    Files are organized by upload timestamp for better tracking.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create timestamp-based directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_dir = os.path.join(
            os.path.dirname(config.PDF_DIR[0] if isinstance(config.PDF_DIR, list) else config.PDF_DIR),
            "uploads",
            timestamp
        )
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        response_data = {
            "success": True,
            "message": "PDF uploaded successfully",
            "filename": file.filename,
            "category": timestamp,  # Use timestamp as identifier
            "file_path": file_path,
            "auto_processed": False
        }
        
        # Auto process if requested
        if auto_process:
            try:
                # Extract text from this specific PDF
                out_dir = os.path.join(config.TXT_DIR, timestamp)
                
                # Extract text (no category-specific logic needed)
                text = extract_text(file_path, skip_first_page=False)
                
                filename = os.path.splitext(file.filename)[0] + ".txt"
                save_text(text, out_dir, filename)
                
                # Run chunk and embed
                chunk_request = ChunkAndEmbedRequest(
                    text_directory=config.TXT_DIR,
                    chunk_size=500,
                    chunk_overlap=50
                )
                await chunk_and_embed_texts(chunk_request)
                
                # Rebuild index
                await build_faiss_index()
                
                response_data["auto_processed"] = True
                response_data["message"] = "PDF uploaded and processed successfully"
                
            except Exception as e:
                response_data["message"] = f"PDF uploaded but processing failed: {str(e)}"
        
        return UploadPDFResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@router.post("/extract-text", response_model=dict)
async def extract_pdf_text():
    """
    Extract text from PDF files in the configured directories.
    
    This endpoint processes all PDF files from the directories specified in config
    and saves them as text files.
    """
    try:
        pdf_dirs = config.PDF_DIR
        root_out_dir = config.TXT_DIR
        total_files = 0
        
        for pdf_dir in pdf_dirs:
            if not os.path.exists(pdf_dir):
                continue
                
            category = os.path.basename(os.path.normpath(pdf_dir))
            out_dir = os.path.join(root_out_dir, category)
            skip_first = category in ["Computer", "Physics"]
            
            for file in os.listdir(pdf_dir):
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(pdf_dir, file)
                    print(f"Extracting: {pdf_path}")
                    
                    # Use OCR for Physics, regular extraction for others
                    if category == "Physics":
                        text = extract_text_ocr(pdf_path, skip_first_page=skip_first)
                    else:
                        text = extract_text(pdf_path, skip_first_page=skip_first)
                    
                    filename = os.path.splitext(file)[0] + ".txt"
                    save_text(text, out_dir, filename)
                    total_files += 1
        
        return {
            "success": True,
            "message": "Text extraction completed successfully",
            "total_files_processed": total_files,
            "output_directory": root_out_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")


@router.post("/full-pipeline", response_model=dict)
async def full_pipeline(request: ChunkAndEmbedRequest):
    """
    Execute the full RAG pipeline: text extraction -> chunking -> embedding -> index building.
    
    This endpoint runs all steps sequentially to create a complete FAISS index
    from PDF files.
    """
    try:
        # Step 1: Extract text from PDFs
        extraction_response = await extract_pdf_text()
        
        # Step 2: Chunk and embed
        chunk_embed_response = await chunk_and_embed_texts(request)
        
        # Step 3: Build index
        build_response = await build_faiss_index()
        
        return {
            "success": True,
            "message": "Full RAG pipeline completed successfully",
            "text_extraction": {
                "total_files_processed": extraction_response["total_files_processed"],
                "output_directory": extraction_response["output_directory"]
            },
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
