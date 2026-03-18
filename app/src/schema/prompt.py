"""
Prompt templates for dialogue generation
"""

def get_dialogue_prompt_with_context(context: str, instruction: str, difficulty: str) -> str:
    """
    Generate prompt for dialogue generation with RAG context
    
    Args:
        context: Retrieved context from RAG
        instruction: User's instruction for dialogue generation
        difficulty: Difficulty level (easy, medium, hard, professional)
    
    Returns:
        Formatted prompt string
    """
    return f"""Use the information provided in the relevant context to generate a natural dialogue between two people (Person A and Person B).

### Relevant Context:
{context}

### Instruction:
{instruction}

context difficulty: {difficulty}

### Requirements:
- Format each line as "A: [text]" or "B: [text]"
- Make it conversational and natural
- Keep responses focused on the context provided
- Base on the context difficulty level (easy, medium, hard, professional)

### Dialogue:"""


def get_dialogue_prompt_without_context(instruction: str) -> str:
    """
    Generate prompt for dialogue generation without RAG context
    
    Args:
        instruction: User's instruction for dialogue generation
    
    Returns:
        Formatted prompt string
    """
    return f"""Generate a natural dialogue between two people (Person A and Person B) based on the following instruction.

### Instruction:
{instruction}

### Requirements:
- Format each line as "A: [text]" or "B: [text]"
- Make it conversational and natural

### Dialogue:"""


DIALOGUE_SYSTEM_PROMPT = "You are a helpful assistant that generates natural dialogues between two people based on given context."
