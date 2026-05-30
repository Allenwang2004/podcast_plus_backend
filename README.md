## Podcast+ Backend

A personal podcast generation system that transforms static knowledge into interactive, personalized audio content using RAG (Retrieval-Augmented Generation), dialogue generation, and TTS (Text-to-Speech) technologies.

---

## System Architecture Overview

### Vocal Insertion Pipeline

1. User records audio
2. Frontend converts audio format
3. POST to Next.js API Route
4. Whisper STT (Speech-to-Text)
5. Pass transcribed text as user_instruction

### TTS (Text-to-Speech) Pipeline

#### Process Flow
1. Frontend receives generate dialogue and audioid
2. Call generate_audio with dialogue and audioid
3. Backend parses the dialogue
4. Use TTS to generate audio segments
5. Merge audio segments
6. Save to server
7. Frontend plays audio via URL generated from audioid

#### Architecture Design

Always-running service + File queue communication

```
FastAPI Backend              Audio Worker Service       Retrieval Worker Service
     |                              |                           |
     |                              |                           | (Always running)
     |                              |                           | Models pre-loaded
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
     |                                                          | Perform retrieval
     |                                                          | (FAISS + Embedding)
     |                                                          |
     |<--  retrieve_result.json   ------------------------------|
     |
     | Return audio URL or retrieval results
```

**Advantages:**
- ✅ Models loaded only once, speed increased 5-10x
- ✅ Parallel audio generation, speedup 2-4x
- ✅ Support multiple concurrent requests
- ✅ Service failures don't affect main process


### TTS Voice Settings

#### American English

**Female Voices:**
- **af_heart**: Lively and energetic
- **af_bella**: Gentle and warm
- **af_nicole**: Meditation style
- **af_sarah**: Neutral/Standard
- **af_sky**: Neutral/Standard
- **af_alloy**: Neutral/Standard
- **af_aoede**: Neutral/Standard
- **af_kore**: Neutral/Standard
- **af_nova**: Neutral/Standard
- **af_river**: Neutral/Standard
- **af_jessica**: Neutral/Standard

**Male Voices:**
- **am_adam**: Neutral/Standard
- **am_echo**: Gentle voice
- **am_eric**: Neutral/Standard
- **am_fenrir**: Lively and energetic
- **am_liam**: Neutral/Standard
- **am_michael**: Meditation style
- **am_onyx**: Neutral/Standard
- **am_puck**: Neutral/Standard

#### British English

**Female Voices:**
- **bf_alice**: Neutral/Standard
- **bf_emma**: Neutral/Standard
- **bf_isabella**: Neutral/Standard
- **bf_lily**: British accent

**Male Voices:**
- **bm_daniel**: Neutral/Standard
- **bm_fable**: Neutral/Standard
- **bm_george**: Neutral/Standard
- **bm_lewis**: British accent

---

## Project Progress and Status

### Completed Features ✅

- [x] RAG pipeline implementation
- [x] PDF file extraction pipeline
- [x] Upload pipeline: Store uploaded data in backend after compression, trigger RAG pipeline to generate index
- [x] Frontend voice input integration
- [x] TTS pipeline: Retrieve relevant content, generate dialogue with LLM, produce audio files, return to frontend
- [x] Retrieval pipeline
- [x] Docker containerization
- [x] Production deployment: Backend on Digital Ocean, Frontend on Vercel
- [x] Multi-format data extraction support
- [x] Web search integration
- [x] Personalization settings: Voice selection via Kokoro TTS, dialogue difficulty adjustment via prompt
- [x] Search function integration: Function-based rather than API, disabled RAG when web search is selected
- [x] Demo examples and use cases
- [x] Memory functionality: LLM-as-a-judge to determine conversation continuity, reuse previous retrieval context for follow-up questions
- [x] Retrieval worker service

### In Development / Planned Features 🚀

- [ ] Agent debate system
- [ ] Knowledge base optimization: Classify uploaded PDFs by embedding, select representative values for each category to improve retrieval efficiency
- [ ] Low-confidence filter: Skip retrieval information if confidence score is below threshold

---

## Project Overview and Purpose

### Primary Objective

**Problem Statement:** While podcasts are popular for knowledge acquisition, they are created by content creators. This project provides a **user-controlled podcast generation system** that empowers users to create personalized podcasts from their own knowledge sources.

### System Architecture

The system combines:
- **Frontend-Backend Configuration**: Modern web stack with separated concerns
- **Worker Design Pattern**: Handles concurrent processing despite single-server constraint
- **RAG Integration**: Leverages existing knowledge bases for context-aware generation
- **Real-time Dialogue**: Support for interactive conversations with memory

### Use Cases

1. **Knowledge Base Scenario**: Generate podcast episodes from uploaded documents (e.g., F1 racing rules extraction)
2. **Web Search Scenario**: Create dynamic content from web searches with real-time generation (e.g., world business news)
3. **Conversational Scenario**: Interactive voice-based dialogue with context memory (e.g., Taiwan-related topics)

### Technical Excellence

- **Original Architecture**: Novel combination of RAG, dialogue generation, and TTS technologies beyond traditional one-way content delivery
- **Engineering Quality**: Clean system architecture with clear data flow logic
- **Deep Integration**: Demonstrates proficient model selection and technology integration
- **Production Ready**: Complete implementation from concept to deployment

### Evaluation Criteria

- **Design Concept (40%)**: Originality and practical application potential
- **Completeness (20%)**: Feature completeness and quality (stability and performance)
- **Presentation (20%)**: Documentation, video content, and demo quality
- **Professional Practice (15%)**: Software engineering methodology and development tools
- **AI Integration (5%)**: Effective use of AI throughout the project lifecycle

---

*Last Updated: May 30, 2026*