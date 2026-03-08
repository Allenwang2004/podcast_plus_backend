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
    IndexStatusResponse,
    UploadPDFResponse
)
from rag.chunking import chunk_texts
from rag.embedding import embed_chunks
from rag.build_index import build_index
from rag.text_extraction import extract_all_text, save_text

router = APIRouter()
config = Config()


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
                # Extract text from this specific PDF using new method
                out_dir = os.path.join(config.TXT_DIR, timestamp)
                
                # Use new pdfplumber-based extraction with auto blacklist
                text = extract_all_text(file_path, skip_first_page=False)
                
                filename = os.path.splitext(file.filename)[0] + ".txt"
                save_text(text, out_dir, filename)
                
                # Run chunk
                chunk_texts(
                    text_dir=config.TXT_DIR,
                    chunk_size=500,
                    chunk_overlap=50
                )
                
                # Run embed
                embed_chunks(
                    chunk_dir=config.CHUNK_DIR,
                    embedding_model=None
                )
                
                # Rebuild index
                build_index()
                
                response_data["auto_processed"] = True
                response_data["message"] = "PDF uploaded and processed successfully"
                
            except Exception as e:
                response_data["message"] = f"PDF uploaded but processing failed: {str(e)}"
        
        return UploadPDFResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


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
