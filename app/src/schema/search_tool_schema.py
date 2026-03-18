from pydantic import BaseModel, Field
from typing import Optional, List

class SearchToolRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Title of the podcast episode to search for")

class SearchToolResponse(BaseModel):
    success: bool
    results: List[str] = Field(default_factory=list, description="List of search results (e.g. episode titles or links)")
    message: str