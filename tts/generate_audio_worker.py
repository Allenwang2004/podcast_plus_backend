#!/usr/bin/env python3
"""
Standalone TTS audio generation worker
Run TTS in a separate process to avoid crashes in main FastAPI server
"""
import sys
import json
import re
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings
from kokoro import KPipeline

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils.weight_norm')

def generate_audio(dialogue: str, audio_id: str, output_dir: str):
    """Generate audio from dialogue text"""
    try:
        print(f"[Worker] Starting audio generation for: {audio_id}", file=sys.stderr)
        
        # Initialize TTS pipeline
        print("[Worker] Loading Kokoro TTS pipeline...", file=sys.stderr)
        pipeline = KPipeline(
            lang_code='a',
            device='cpu',
            repo_id='hexgrad/Kokoro-82M'
        )
        print("[Worker] TTS pipeline loaded", file=sys.stderr)
        
        # Parse dialogue
        lines = dialogue.strip().split('\n')
        audio_segments = []
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Parse speaker and text
            match = re.match(r'^([AB]):\s*(.+)$', line)
            if not match:
                print(f"[Worker] Skipping invalid line: {line[:50]}...", file=sys.stderr)
                continue
                
            speaker, text = match.groups()
            voice = "af_heart" if speaker == "A" else "am_adam"
            
            print(f"[Worker] Processing line {idx+1}: Speaker {speaker}", file=sys.stderr)
            
            # Generate audio
            generator = pipeline(text, voice=voice)
            for _, _, audio in generator:
                audio_segments.append(audio)
                break
        
        if not audio_segments:
            raise ValueError("No valid dialogue lines found")
        
        # Concatenate with silence
        print(f"[Worker] Concatenating {len(audio_segments)} segments...", file=sys.stderr)
        silence = np.zeros(int(24000 * 0.3))
        final_audio = []
        for i, segment in enumerate(audio_segments):
            final_audio.append(segment)
            if i < len(audio_segments) - 1:
                final_audio.append(silence)
        
        final_audio = np.concatenate(final_audio)
        
        # Save audio
        output_path = Path(output_dir) / f"{audio_id}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), final_audio, 24000)
        
        print(f"[Worker] Audio saved to: {output_path}", file=sys.stderr)
        
        # Return success
        result = {
            "success": True,
            "output_path": str(output_path),
            "message": "Audio generated successfully"
        }
        print(json.dumps(result))
        return 0
        
    except Exception as e:
        print(f"[Worker] Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(result))
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: generate_audio_worker.py <dialogue> <audio_id> <output_dir>", file=sys.stderr)
        sys.exit(1)
    
    dialogue = sys.argv[1]
    audio_id = sys.argv[2]
    output_dir = sys.argv[3]
    
    sys.exit(generate_audio(dialogue, audio_id, output_dir))
