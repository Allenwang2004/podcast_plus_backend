# Podcast+

作品名稱：Podcast+

1. 作品以行動情境下的資訊吸收出發，將靜態知識轉化為可互動的 Podcast，不僅應用場景明確，更透過個人化收聽與即時互動設計，賦予內容生成工具極高的實用價值與落地潛力。在原創性上，團隊成功將 RAG、對話生成與 TTS 技術重新組構，跳脫傳統單向輸出的框架，展現出具備差異化的創新思考。程式專業性表現尤為紮實，系統架構與資料流程邏輯清晰，顯見團隊對模型選用與技術整合具備深度的掌握力。整體作品從概念發想到實作展示皆完整且成熟，影片解說條理分明，充分體現了技術細節與應用情境的對應關係。若未來能進一步導入實際使用者測試以驗證情境案例，將使該方案在解決現實問題的說服力上更臻完善，是極具延伸發展價值的專題佳作。
2. 系統整合不少模組，具備良好工程性，完成性高。但是本質上似乎就是利用使用者輸入的prompt，讓大語言模型產出對應的內容，再以TTS模組轉換成語音，是否真能達成針對使用者需求製作podcast之目的？

### TODO

- [x]  rag pipeline
- [x]  extracting pipline for pdf file
- [x]  upload pipleine: 上傳資料後，壓縮後存放到後端資料夾裡，啟動 rag pipline 後存放 index
- [x]  frontend vocal insert: 語音輸入
- [x]  tts-pipline: 啟動 retrieval 後，根據檢索結果，給模型產出結果，產出音檔，並回傳給前端(看是要直接做音訊處理還是要用url的形式)
- [x]  retrieval pipline
- [x]  container
- [x]  production : backend 部署在 digital ocean 上 前端部署在 vercel
- [x]  multi data-form extraction
- [x]  Search tool
- [x]  個人化設定 : 聲音選擇 by kokoro model 還有對話難度設定 by prompt
- [x]  search tool 不用當作 API 應該做為一個 function，如果前端網頁收尋功能有被選取，就會使用 search_function 當作檢索到的資訊，use_rag == false
- [x]  demo 例子呈現

- [x]  後端要有 memory 的功能，用 LLM-as-a-judge 判斷問句是否和上一段產出有關，追加問題的話，把上一次檢索內容再餵一次
- [x]  檢索 retrieve worker
- [ ]  Agent debate system
- [ ]  knowledge base 將 upload 的 pdf 根據 embedding 分類，針對每個分類取一個值作為代表，提升檢索效率
- [ ]  如果檢索信心不高，不要使用檢索資訊

### 報告內容
* 產品目的 : 聆聽 podcast 是很多人獲取新知識的一個新方法，但因為 podcast 本身是由創作者所做，我們想做的就是一個完全由個人掌握的一個 podcast 產生系統
* 系統大致架構 : 前後端的配置，然後提到 worker 的設計(因為使用單一 server)
* 使用情境一 : 根據知識庫內的產出，就用 F1 的例子，可以把原本的規則截圖下來，這個直接截圖說明幾可
* 使用情境二 : 根據網頁收尋的產出，用打字輸入的，但就要等他產出(把 RAG 關掉會快)，這段時間跳回來說明可以來講如何做網頁收尋，這應該就用 WBC 的例子
* 使用情境三 : 根據即時對話有 memory 的產出，用語音輸入，產出的時候跳回來講怎麼處理上下文記憶跟未來展望，例子就是說想聽到跟台灣相關的

### 評分標準
* 40% 設計概念：包含原創性及應用性
* 20% 完成性：實作之作品功能是否完整，品質(如穩定度與效能)是否良好。
* 20% 報告與展示：包含決賽文件與影片內容、現場簡報與問答。
* 15% 專業性：程式寫作過程是否結合運算思維和軟體開發方法與工具。
* 5%  AI應用：從需求、設計、實作到測試，妥善應用AI協助。


### Vocal insertion pipline

1. 用戶錄音
2. 前端轉換格式
3. POST 到 Next.js API Route
4. Whisper STT
5. 將文字作為 user_instruction

### TTS pipline

#### 流程說明
1. 前端接受 generate dialogue 和 audioid
2. 使用 generate_audio 傳入 dialogue 和 audioid
3. 後端解析對話
4. 使用 TTS 生成音頻
5. 合併音頻片段
6. 存到服務器
7. 前端通過 audioid 產生的 URL 播放

#### 架構設計
常駐服務 + 文件隊列通信

```
FastAPI Backend              Audio Worker Service       Retrieval Worker Service
     |                              |                           |
     |                              |                           | (持續運行)
     |                              |                           | 模型已加載
     |                              |                           |
     |----- task.json ------------> |                           |
     |                              |                           |
     |                              |                           |
     |                              |                           |
     |                              |                           |
     |                              |                           |
     |<----- result.json -----------|                           |
     |                              |                           |
     |------------------ retrieve_task.json ------------------->|
     |                                                          |
     |                                                          | 執行檢索
     |                                                          | (FAISS + Embedding)
     |                                                          |
     |<--  retrieve_result.json   ------------------------------|
     |
     | 回傳音頻 URL 或檢索結果
```

**優勢：**
- ✅ 模型只加載一次，速度提升 5-10 倍
- ✅ 音頻並行生成，加速 2-4 倍
- ✅ 支援多個請求同時處理
- ✅ 服務異常不影響主進程


### TTS voice setting
+ For American women:
  + af_heart：lively
  + af_bella：gentle
  + af_nicole meditation
  + af_sarah : pass
  + af_sky : pass
  + af_alloy : pass
  + af_aoede : pass
  + af_kore : pass
  + af_nova : pass
  + af_river : pass
  + af_jessica : pass
+ For American men:
  + am_adam : pass
  + am_echo gentle
  + am_eric : pass
  + am_fenrir：lively
  + am_liam : pass
  + am_michael : meditation
  + am_onyx : pass
  + am_puck : pass
+ For British women:
  + bf_alice : pass
  + bf_emma : pass
  + bf_isabella :pass
  + bf_lily british
+ For British men:
  + bm_daniel : pass
  + bm_fable : pass
  + bm_george : pass
  + bm_lewis : british