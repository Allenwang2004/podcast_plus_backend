import os
import trafilatura
from fastapi import APIRouter, HTTPException
from tavily import TavilyClient
from app.src.schema.search_tool_schema import SearchToolRequest, SearchToolResponse, SearchResultItem

router = APIRouter()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@router.post("/web-search", response_model=SearchToolResponse)
async def web_search(request: SearchToolRequest):
    """
    執行即時網路搜尋，抓取網頁內容並回傳。
    """
    if not TAVILY_API_KEY:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY not set")

    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        
        # 1. 執行搜尋
        print(f"[Search] Searching web for: {request.query}")
        search_data = tavily.search(
            query=request.query, 
            search_depth="advanced", 
            max_results=request.max_results
        )
        
        final_results = []
        
        # 2. 遍歷結果並使用 trafilatura 抓取更完整的內文
        for item in search_data.get("results", []):
            url = item.get("url")
            
            downloaded = trafilatura.fetch_url(url)
            full_content = trafilatura.extract(downloaded, include_comments=False)
            
            content = full_content if full_content else item.get("content", "")
            
            final_results.append(SearchResultItem(
                title=item.get("title", "No Title"),
                url=url,
                content=content[:2000],
                score=item.get("score", 0.0)
            ))

        if not final_results:
            return SearchToolResponse(success=True, results=[], message="查無相關網路資訊")

        return SearchToolResponse(
            success=True, 
            results=final_results, 
            message=f"成功找到 {len(final_results)} 筆即時資訊"
        )

    except Exception as e:
        print(f"[Search Error] {str(e)}")
        return SearchToolResponse(success=False, results=[], message=f"搜尋失敗: {str(e)}")