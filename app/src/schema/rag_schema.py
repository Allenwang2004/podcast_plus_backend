from pydantic import BaseModel, Field
from typing import Optional, List

class BuildIndexRequest(BaseModel):
    """Request model for building FAISS index"""
    chunk_directory: Optional[str] = Field(
        default=None, 
        description="Directory containing chunk JSON files. If not provided, uses default from config."
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model name. If not provided, uses default from config."
    )
    
class BuildIndexResponse(BaseModel):
    """Response model for building FAISS index"""
    success: bool
    message: str
    index_path: str
    num_vectors: int
    dimension: int
    
class ChunkAndEmbedRequest(BaseModel):
    """Request model for chunking and embedding texts"""
    text_directory: Optional[str] = Field(
        default=None,
        description="Directory containing text files. If not provided, uses default from config."
    )
    chunk_size: Optional[int] = Field(default=500, description="Size of text chunks")
    chunk_overlap: Optional[int] = Field(default=50, description="Overlap between chunks")
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model name. If not provided, uses default from config."
    )

class ChunkAndEmbedResponse(BaseModel):
    """Response model for chunking and embedding"""
    success: bool
    message: str
    num_chunks: int
    num_embeddings: int
    embedding_shape: List[int]

class IndexStatusResponse(BaseModel):
    """Response model for index status"""
    exists: bool
    path: str
    num_vectors: Optional[int] = None
    dimension: Optional[int] = None
    message: str
