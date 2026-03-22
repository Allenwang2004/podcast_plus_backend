from fastapi import APIRouter, HTTPException
from openai import OpenAI
import os
import uuid
import re
import subprocess
import json
from pathlib import Path
from app.src.schema.podcast_schema import (
    GenerateDialogueRequest, 
    GenerateDialogueResponse,
    GenerateAudioRequest,
    GenerateAudioResponse
)
from app.src.schema.prompt import (
    get_dialogue_prompt_with_context,
    get_dialogue_prompt_without_context,
    get_dialogue_prompt_with_web_search,
    DIALOGUE_SYSTEM_PROMPT
)

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import Config

router = APIRouter()
config = Config()

@router.post("/generate-dialogue", response_model=GenerateDialogueResponse)
async def generate_dialogue(request: GenerateDialogueRequest):
    """
    Generate a natural dialogue between two people based on user instruction and optional context.
    
    - **user_instruction**: The instruction for dialogue generation
    - **retrieved_context**: Optional pre-retrieved context. If not provided, will auto-retrieve from RAG if use_rag=True
    - **use_rag**: Whether to automatically retrieve context from RAG system (default: True)
    - **top_n**: Number of chunks to retrieve from RAG (default: 3)
    - **difficulty**: Difficulty level for dialogue generation (easy, medium, hard) (default: medium)
    - **model**: OpenAI model to use (default: gpt-4o-mini)
    - **max_tokens**: Maximum tokens for generation (default: 1000)
    """
    try:
        # Check if OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail="OPENAI_API_KEY environment variable is not set. Please set it before using this endpoint."
            )
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Get or retrieve context
        context_to_use = request.retrieved_context
        
        # If web_search_context is provided, skip RAG retrieval
        if request.web_search_context:
            print("[Web Search] Using provided web search context, skipping RAG retrieval")
        elif not context_to_use and request.use_rag:
            try:
                print(f"[RAG] Retrieving context for: {request.user_instruction[:50]}...")
                
                # Check if FAISS index exists
                if not os.path.exists(config.FAISS_INDEX):
                    print("[RAG] Warning: FAISS index not found, generating without context")
                else:
                    # Use queue-based retrieval service
                    queue_dir = Path(__file__).parent.parent.parent.parent / "worker" / "queue"
                    queue_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate unique task ID
                    task_id = str(uuid.uuid4())
                    
                    # Create retrieval task
                    task = {
                        "task_id": task_id,
                        "query": request.user_instruction,
                        "top_n": request.top_n
                    }
                    
                    task_file = queue_dir / f"retrieve_task_{task_id}.json"
                    with open(task_file, 'w', encoding='utf-8') as f:
                        json.dump(task, f)
                    
                    print(f"[RAG] Task submitted to retrieval service")
                    
                    # Wait for result (with timeout)
                    result_file = queue_dir / f"retrieve_result_{task_id}.json"
                    timeout = 30  # 30 seconds timeout
                    start_time = Path(task_file).stat().st_mtime
                    
                    import time
                    while True:
                        if result_file.exists():
                            # Read result
                            with open(result_file, 'r', encoding='utf-8') as f:
                                worker_result = json.load(f)
                            
                            # Clean up result file
                            result_file.unlink()
                            
                            if worker_result.get("success"):
                                context_to_use = worker_result.get("context", "")
                                num_chunks = worker_result.get("num_chunks", 0)
                                if context_to_use:
                                    print(f"[RAG] Retrieved {num_chunks} chunks via service")
                                else:
                                    print("[RAG] No relevant chunks found")
                            else:
                                print(f"[RAG] Service failed: {worker_result.get('error', 'Unknown error')}")
                            break
                        
                        # Check timeout
                        elapsed = time.time() - start_time
                        if elapsed > timeout:
                            # Clean up task file if still exists
                            if task_file.exists():
                                task_file.unlink()
                            print("[RAG] Retrieval timeout, generating without context")
                            break
                        
                        # Sleep briefly before checking again
                        time.sleep(0.1)
                        
            except Exception as e:
                print(f"[RAG] Retrieval failed: {str(e)}, generating without context")
                # Continue without context rather than failing
        
        # Build prompt based on available contexts
        if request.web_search_context:
            # Use web search context (with or without RAG context)
            if context_to_use:
                # Both web search and RAG context available
                prompt = get_dialogue_prompt_with_web_search(
                    context=context_to_use,
                    instruction=request.user_instruction,
                    difficulty=request.difficulty,
                    web_search_context=request.web_search_context
                )
            else:
                # Only web search context available, treat it as main context
                prompt = get_dialogue_prompt_with_context(
                    context=request.web_search_context,
                    instruction=request.user_instruction,
                    difficulty=request.difficulty
                )
        elif context_to_use:
            # Only RAG context available
            prompt = get_dialogue_prompt_with_context(
                context=context_to_use,
                instruction=request.user_instruction,
                difficulty=request.difficulty
            )
        else:
            # No context available
            prompt = get_dialogue_prompt_without_context(
                instruction=request.user_instruction
            )
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": DIALOGUE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=request.max_tokens,
            temperature=0.8,
            top_p=0.9
        )
        
        dialogue = response.choices[0].message.content
        
        # Generate unique audio ID
        audio_id = str(uuid.uuid4())
        
        return GenerateDialogueResponse(
            success=True,
            dialogue=dialogue,
            audio_id=audio_id,
            message="Dialogue generated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dialogue: {str(e)}")

@router.post("/generate-audio", response_model=GenerateAudioResponse)
def generate_audio(request: GenerateAudioRequest):
    """
    Generate audio from dialogue text and save it with the given audio_id.
    Uses a persistent worker service for faster processing (no model reload).
    
    - **dialogue**: The dialogue text in format "A: text\nB: text"
    - **voice_type**: Voice type for audio generation (default: "gentle")
    - **audio_id**: Unique ID for the audio file
    """
    try:
        print(f"[Audio Generation] Starting for audio_id: {request.audio_id}")
        
        # Create static audio directory if not exists
        static_dir = Path(config.STATIC_DIR) / "audio"
        static_dir.mkdir(parents=True, exist_ok=True)
        
        # Queue directory for worker service
        queue_dir = Path(__file__).parent.parent.parent.parent / "worker" / "queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        
        # Create task file
        task = {
            "dialogue": request.dialogue,
            "voice_type": request.voice_type,
            "audio_id": request.audio_id,
            "output_dir": str(static_dir)
        }
        
        task_file = queue_dir / f"task_{request.audio_id}.json"
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task, f)
        
        print(f"[Audio Generation] Task submitted to queue")
        
        # Wait for result (with timeout)
        result_file = queue_dir / f"result_{request.audio_id}.json"
        timeout = 300  # 5 minutes
        start_time = Path(task_file).stat().st_mtime
        
        while True:
            if result_file.exists():
                # Read result
                with open(result_file, 'r', encoding='utf-8') as f:
                    worker_result = json.load(f)
                
                # Clean up result file
                result_file.unlink()
                
                if not worker_result.get("success"):
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Audio generation failed: {worker_result.get('error', 'Unknown error')}"
                    )
                
                # Generate URL
                audio_url = f"{config.BASE_URL}/static/audio/{request.audio_id}.wav"                
                print(f"[Audio Generation] BASE_URL: {config.BASE_URL}", file=sys.stderr)
                print(f"[Audio Generation] Generated URL: {audio_url}", file=sys.stderr)                
                print(f"[Audio Generation] Successfully generated audio: {audio_url}")
                
                return GenerateAudioResponse(
                    success=True,
                    audio_url=audio_url,
                    message="Audio generated successfully"
                )
            
            # Check timeout
            import time
            elapsed = time.time() - start_time
            if elapsed > timeout:
                # Clean up task file if still exists
                if task_file.exists():
                    task_file.unlink()
                raise HTTPException(
                    status_code=504, 
                    detail="Audio generation timed out. Make sure the worker service is running."
                )
            
            # Sleep briefly before checking again
            time.sleep(0.2)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Audio Generation] Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")