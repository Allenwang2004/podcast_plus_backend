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
    執行即時網路搜尋，抓取網頁內容，分開標題與內文並回傳。
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
        context_chunks = []
        
        # 2. 遍歷結果並處理內容
        for item in search_data.get("results", []):
            url = item.get("url")
            title = item.get("title", "No Title")
            
            # 抓取並清理內文
            downloaded = trafilatura.fetch_url(url)
            full_content = trafilatura.extract(downloaded, include_comments=False)
            
            # 優先使用 trafilatura 抓取的內容，若無則回退到 Tavily 摘要
            content = (full_content if full_content else item.get("content", ""))[:2000]

            # A. 存入結構化清單 (供前端使用)
            final_results.append(SearchResultItem(
                title=title,
                url=url,
                content=content,
                score=item.get("score", 0.0)
            ))

            # B. 建立給 LLM 讀取的乾淨格式
            # 使用明確的標籤讓 LLM 區分 Title 與 Content
            formatted_item = f"SOURCE TITLE: {title}\nSOURCE CONTENT: {content}"
            context_chunks.append(formatted_item)

        if not final_results:
            return SearchToolResponse(
                success=True, 
                results=[], 
                formatted_context="", 
                message="查無相關網路資訊"
            )

        # 3. 將所有結果用分隔線連接，限制總長度防止爆 Token
        # 分隔線 --- 能有效幫助 LLM 區分不同網頁來源
        all_context = "\n\n---\n\n".join(context_chunks)[:6000]

        return SearchToolResponse(
            success=True, 
            results=final_results, 
            formatted_context=all_context, # 這個欄位直接丟給後續的 LLM 生成
            message=f"成功找到 {len(final_results)} 筆即時資訊"
        )

    except Exception as e:
        print(f"[Search Error] {str(e)}")
        return SearchToolResponse(success=False, results=[], formatted_context="", message=f"搜尋失敗: {str(e)}")