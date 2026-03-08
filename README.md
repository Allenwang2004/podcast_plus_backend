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
- [ ]  retrieval pipline
- [ ]  user interaction : 與 podcast 進行 realtime 互動
- [ ]  container
- [ ]  production : backend 部署在 render 上 前端部署在 vercel
- [ ]  效能優化

### Vocal insertion pipline

用戶錄音 → 前端轉換格式 → POST 到 Next.js API Route 
→ Whisper STT → 將文字作為 user_instruction

### TTS pipline
前端接受 generate dialogue 和 audioid 後 → 使用 generate_audio 傳入 dialogue 和 audioid -> 後端解析對話 
→ 使用 TTS 生成音頻 → 合併音頻片段 → 存到服務器 → 前端通過 audioid 產生的 URL 播放

分離進程，使用 subprocess

FastAPI main process           Worker process
     |                            |
     |-- subprocess.run() ------->|
     |                            | 加載 Kokoro
     |                            | 生成音頻
     |                            | 保存文件
     |<---- JSON 结果 ------------|
     |                            | 退出 process
     | 解析结果
     | 返回 URL 给前端