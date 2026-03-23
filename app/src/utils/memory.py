"""
Memory management utilities for storing conversation history
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

def save_conversation_memory(
    instruction: str,
    dialogue: str,
    audio_id: str,
    context: Optional[str] = None,
    web_search_context: Optional[str] = None,
    difficulty: str = "medium",
    memory_dir: str = "db/memory"
):
    """
    Save conversation to memory as JSON file
    
    Args:
        instruction: User's instruction
        dialogue: Generated dialogue
        audio_id: Unique audio ID
        context: RAG context (optional)
        web_search_context: Web search context (optional)
        difficulty: Difficulty level
        memory_dir: Directory to save memory files
    """
    # Create memory directory if not exists
    memory_path = Path(memory_dir)
    memory_path.mkdir(parents=True, exist_ok=True)
    
    # Create memory entry
    memory_entry = {
        "audio_id": audio_id,
        "timestamp": datetime.now().isoformat(),
        "instruction": instruction,
        "dialogue": dialogue,
        "context": context,
        "web_search_context": web_search_context,
        "difficulty": difficulty
    }
    
    # Save to file (using audio_id as filename)
    file_path = memory_path / f"{audio_id}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(memory_entry, f, ensure_ascii=False, indent=2)
    
    return str(file_path)


def get_conversation_memory(audio_id: str, memory_dir: str = "db/memory") -> Optional[dict]:
    """
    Retrieve conversation from memory by audio_id
    
    Args:
        audio_id: Unique audio ID
        memory_dir: Directory where memory files are stored
    
    Returns:
        Memory entry dict or None if not found
    """
    file_path = Path(memory_dir) / f"{audio_id}.json"
    
    if not file_path.exists():
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_recent_conversations(limit: int = 10, memory_dir: str = "db/memory") -> list:
    """
    Get recent conversations from memory
    
    Args:
        limit: Maximum number of conversations to return
        memory_dir: Directory where memory files are stored
    
    Returns:
        List of memory entries, sorted by timestamp (newest first)
    """
    memory_path = Path(memory_dir)
    
    if not memory_path.exists():
        return []
    
    # Get all JSON files
    memory_files = list(memory_path.glob("*.json"))
    
    # Load and sort by timestamp
    memories = []
    for file_path in memory_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                memory = json.load(f)
                memories.append(memory)
        except Exception as e:
            print(f"Error loading memory file {file_path}: {e}")
            continue
    
    # Sort by timestamp (newest first)
    memories.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return memories[:limit]
