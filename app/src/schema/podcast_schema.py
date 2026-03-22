from pydantic import BaseModel, Field
from typing import Optional

class GenerateDialogueRequest(BaseModel):
    """Request model for generating dialogue"""
    user_instruction: str = Field(..., description="User's instruction for dialogue generation")
    retrieved_context: Optional[str] = Field(default=None, description="Optional pre-retrieved context. If not provided and use_rag=True, will auto-retrieve from RAG.")
    use_rag: bool = Field(default=True, description="Whether to use RAG for automatic context retrieval")
    difficulty: Optional[str] = Field(default="medium", description="Difficulty level for dialogue generation (easy, medium, hard)")
    top_n: Optional[int] = Field(default=3, description="Number of chunks to retrieve from RAG")
    model: Optional[str] = Field(default="gpt-4o-mini", description="OpenAI model to use")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens for generation")
    web_search_context: Optional[str] = Field(default=None, description="Optional web search context to include in generation")

class GenerateDialogueResponse(BaseModel):
    """Response model for generated dialogue"""
    success: bool
    dialogue: str
    audio_id: str
    message: str

class GenerateAudioRequest(BaseModel):
    """Request model for generating audio from dialogue"""
    dialogue: str = Field(..., description="Dialogue text to convert to audio")
    voice_type: Optional[str] = Field(default="gentle", description="Voice type for audio generation")
    audio_id: str = Field(..., description="Unique ID for the audio file")

class GenerateAudioResponse(BaseModel):
    """Response model for generated audio"""
    success: bool
    audio_url: str
    message: str
