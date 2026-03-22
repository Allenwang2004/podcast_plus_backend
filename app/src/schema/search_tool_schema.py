from pydantic import BaseModel, Field
from typing import Optional, List

class SearchResultItem(BaseModel):
    title: str
    url: str
    content: str
    score: float

class SearchToolRequest(BaseModel):
    query: str = Field(..., description="要搜尋的原始問題或關鍵字")
    max_results: int = Field(default=3, description="回傳的網頁數量")

class SearchToolResponse(BaseModel):
    success: bool
    results: List[SearchResultItem] = Field(default_factory=list)
    formatted_context: str = Field("", description="供 LLM 生成使用的格式化網路資訊")
    message: str