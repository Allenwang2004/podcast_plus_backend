#!/usr/bin/env python3
"""
Persistent TTS audio generation worker service
Keeps the model loaded in memory and processes tasks from a queue
"""
import sys
import json
import re
import time
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings
from kokoro import KPipeline
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils.weight_norm')

class AudioWorkerService:
    def __init__(self, device='cpu'):
        """Initialize worker with TTS model"""
        print(f"[Worker Service] Initializing on {device}...", file=sys.stderr)
        self.pipeline = KPipeline(
            lang_code='a',
            device=device,
            repo_id='hexgrad/Kokoro-82M'
        )
        print("[Worker Service] TTS pipeline loaded and ready", file=sys.stderr)
        self.task_queue = Queue()
        self.is_running = True
        
    def generate_audio(self, dialogue: str, voice_type: str, audio_id: str, output_dir: str):
        """Generate audio from dialogue text"""
        try:
            print(f"[Worker Service] Processing task: {audio_id}", file=sys.stderr)
            
            # Parse dialogue
            lines = dialogue.strip().split('\n')
            dialogue_items = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse speaker and text
                match = re.match(r'^([AB]):\s*(.+)$', line)
                if not match:
                    continue
                    
                speaker, text = match.groups()
                if voice_type == "gentle":
                    voice = "af_bella" if speaker == "A" else "am_echo"
                elif voice_type == "lively":
                    voice = "af_heart" if speaker == "A" else "am_fenrir"
                elif voice_type == "meditation":
                    voice = "af_nicole" if speaker == "A" else "am_michael"
                else:
                    voice = "bf_lily" if speaker == "A" else "bm_lewis"
                
                dialogue_items.append((text, voice))
            
            if not dialogue_items:
                raise ValueError("No valid dialogue lines found")
            
            # Parallel generation function
            def generate_segment(idx_text_voice):
                idx, (text, voice) = idx_text_voice
                generator = self.pipeline(text, voice=voice)
                for _, _, audio in generator:
                    return (idx, audio)
                return None
            
            print(f"[Worker Service] Generating {len(dialogue_items)} segments in parallel...", file=sys.stderr)
            
            # Generate all segments in parallel with proper indexing
            audio_segments = [None] * len(dialogue_items)
            with ThreadPoolExecutor(max_workers=min(4, len(dialogue_items))) as executor:
                future_to_idx = {executor.submit(generate_segment, (idx, item)): idx for idx, item in enumerate(dialogue_items)}
                
                for future in as_completed(future_to_idx):
                    result = future.result()
                    if result:
                        idx, audio = result
                        audio_segments[idx] = audio
            
            # Filter out None values
            audio_segments = [seg for seg in audio_segments if seg is not None]
            
            if not audio_segments:
                raise ValueError("Failed to generate any audio segments")
            
            # Concatenate with silence
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
            
            print(f"[Worker Service] Audio saved: {output_path}", file=sys.stderr)
            
            return {
                "success": True,
                "output_path": str(output_path),
                "message": "Audio generated successfully"
            }
            
        except Exception as e:
            print(f"[Worker Service] Error: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_queue(self):
        """Process tasks from queue"""
        queue_dir = Path(__file__).parent / "queue"
        queue_dir.mkdir(exist_ok=True)
        
        print("[Worker Service] Watching queue directory...", file=sys.stderr)
        
        while self.is_running:
            try:
                # Check for task files
                task_files = sorted(queue_dir.glob("task_*.json"))
                
                if task_files:
                    task_file = task_files[0]
                    
                    try:
                        # Read task
                        with open(task_file, 'r', encoding='utf-8') as f:
                            task = json.load(f)
                        
                        audio_id = task['audio_id']
                        
                        # Process task
                        result = self.generate_audio(
                            dialogue=task['dialogue'],
                            voice_type=task['voice_type'],
                            audio_id=audio_id,
                            output_dir=task['output_dir']
                        )
                        
                        # Write result
                        result_file = queue_dir / f"result_{audio_id}.json"
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f)
                        
                        # Remove task file
                        task_file.unlink()
                        
                    except Exception as e:
                        print(f"[Worker Service] Failed to process {task_file}: {e}", file=sys.stderr)
                        task_file.unlink()
                else:
                    # No tasks, sleep briefly
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n[Worker Service] Shutting down...", file=sys.stderr)
                self.is_running = False
                break
            except Exception as e:
                print(f"[Worker Service] Queue processing error: {e}", file=sys.stderr)
                time.sleep(1)

    def run(self):
        """Start the worker service"""
        print("[Worker Service] Service started", file=sys.stderr)
        self.process_queue()
        print("[Worker Service] Service stopped", file=sys.stderr)

if __name__ == "__main__":
    device = 'cuda' if len(sys.argv) > 1 and sys.argv[1] == '--gpu' else 'cpu'
    service = AudioWorkerService(device=device)
    service.run()
