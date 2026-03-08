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
        
        if not context_to_use and request.use_rag:
            try:
                print(f"[RAG] Retrieving context for: {request.user_instruction[:50]}...")
                
                # Check if FAISS index exists
                if not os.path.exists(config.FAISS_INDEX):
                    print("[RAG] Warning: FAISS index not found, generating without context")
                else:
                    # Path to worker script
                    worker_script = Path(__file__).parent.parent.parent.parent / "worker" / "retrieve_worker.py"
                    
                    if not worker_script.exists():
                        print(f"[RAG] Warning: Worker script not found at {worker_script}")
                    else:
                        # Call worker in subprocess
                        # Don't capture stderr so worker logs appear in terminal
                        result = subprocess.run(
                            ["python", str(worker_script), request.user_instruction, str(request.top_n)],
                            stdout=subprocess.PIPE,
                            stderr=None,  # Let stderr go to terminal
                            text=True,
                            timeout=30  # 30 seconds timeout
                        )
                        
                        if result.returncode == 0:
                            # Parse result from stdout
                            try:
                                worker_result = json.loads(result.stdout.strip().split('\n')[-1])
                                
                                if worker_result.get("success"):
                                    context_to_use = worker_result.get("context", "")
                                    num_chunks = worker_result.get("num_chunks", 0)
                                    if context_to_use:
                                        print(f"[RAG] Retrieved {num_chunks} chunks via worker")
                                    else:
                                        print("[RAG] No relevant chunks found")
                                else:
                                    print(f"[RAG] Worker failed: {worker_result.get('error', 'Unknown error')}")
                            except (json.JSONDecodeError, IndexError) as e:
                                print(f"[RAG] Failed to parse worker output: {str(e)}")
                        else:
                            print(f"[RAG] Worker process failed with code {result.returncode}")
                        
            except subprocess.TimeoutExpired:
                print("[RAG] Retrieval timeout, generating without context")
            except Exception as e:
                print(f"[RAG] Retrieval failed: {str(e)}, generating without context")
                # Continue without context rather than failing
        
        # Build prompt based on whether context is available
        if context_to_use:
            prompt = f"""Use the information provided in the relevant context to generate a natural dialogue between two people (Person A and Person B).

### Relevant Context:
{context_to_use}

### Instruction:
{request.user_instruction}

### Requirements:
- Format each line as "A: [text]" or "B: [text]"
- Make it conversational and natural
- Keep responses focused on the context provided

### Dialogue:"""
        else:
            prompt = f"""Generate a natural dialogue between two people (Person A and Person B) based on the following instruction.

### Instruction:
{request.user_instruction}

### Requirements:
- Format each line as "A: [text]" or "B: [text]"
- Make it conversational and natural

### Dialogue:"""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates natural dialogues between two people based on given context."},
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
    Uses a separate worker process to avoid crashes.
    
    - **dialogue**: The dialogue text in format "A: text\nB: text"
    - **audio_id**: Unique ID for the audio file
    """
    try:
        print(f"[Audio Generation] Starting for audio_id: {request.audio_id}")
        
        # Create static audio directory if not exists
        static_dir = Path(config.STATIC_DIR) / "audio"
        static_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to worker script
        worker_script = Path(__file__).parent.parent.parent.parent / "worker" / "audio_worker.py"
        
        if not worker_script.exists():
            raise HTTPException(status_code=500, detail="Audio worker script not found")
        
        print(f"[Audio Generation] Calling worker process...")
        
        # Call worker in subprocess
        result = subprocess.run(
            ["python", str(worker_script), request.dialogue, request.audio_id, str(static_dir)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        print(f"[Audio Generation] Worker exit code: {result.returncode}")
        print(f"[Audio Generation] Worker stderr: {result.stderr}")
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500, 
                detail=f"Audio generation failed. Check logs for details."
            )
        
        # Parse result from stdout
        try:
            worker_result = json.loads(result.stdout.strip().split('\n')[-1])
        except (json.JSONDecodeError, IndexError) as e:
            print(f"[Audio Generation] Failed to parse worker output: {result.stdout}")
            raise HTTPException(status_code=500, detail="Failed to parse audio generation result")
        
        if not worker_result.get("success"):
            raise HTTPException(
                status_code=500, 
                detail=f"Audio generation failed: {worker_result.get('error', 'Unknown error')}"
            )
        
        # Generate URL
        audio_url = f"http://localhost:8001/static/audio/{request.audio_id}.wav"
        
        print(f"[Audio Generation] Successfully generated audio: {audio_url}")
        
        return GenerateAudioResponse(
            success=True,
            audio_url=audio_url,
            message="Audio generated successfully"
        )
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Audio generation timed out")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Audio Generation] Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")