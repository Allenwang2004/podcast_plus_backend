"""
Prompt templates for dialogue generation
"""

def get_dialogue_prompt_with_context(context: str, instruction: str, difficulty: str, previous_conversation: dict = None) -> str:
    """
    Generate prompt for dialogue generation with RAG context
    
    Args:
        context: Retrieved context from RAG
        instruction: User's instruction for dialogue generation
        difficulty: Difficulty level (easy, medium, hard, professional)
        previous_conversation: Optional previous conversation memory
    
    Returns:
        Formatted prompt string
    """
    previous_context = ""
    if previous_conversation:
        previous_context = f"""\n### Previous Topic (For Reference Only):
{previous_conversation.get('instruction', 'N/A')}

Note: Consider the previous topic ONLY if the current instruction is a follow-up, continuation, or modification request. Otherwise, treat this as a new, independent conversation.\n"""
    
    return f"""Create an engaging podcast-style conversation between two hosts (Person A and Person B) using the information provided.
{previous_context}
### Source Material:
{context}

### Episode Topic/Instruction:
{instruction}

Difficulty Level: {difficulty}

### Podcast Style Guidelines:
- Format each line as "A: [text]" or "B: [text]"
- **IMPORTANT: Generate exactly 5 rounds of dialogue (10 lines total: A speaks, B responds, repeat 5 times)**
- Make it sound like a REAL podcast recording:
  * Use natural speech patterns with filler words (you know, I mean, like, um, right)
  * Include reactions and responses ("Oh really?", "That's fascinating!", "Exactly!", "Wait, what?")
  * Show genuine enthusiasm and curiosity
  * Add personal touches and relatable examples
  * Let hosts interrupt, agree, or build on each other's points naturally
  * Use contractions (don't, it's, we're) for casual tone
  * Don't need the opening remark
- Keep it conversational but informative
- Add some humor or personality where appropriate
- If continuing a previous conversation, for example, the instruction says "No, I want to know more about...", generate responses with first sentences like "Well, speaking of that...", "Actually, that reminds me...", or "Building on what you just said..."
- Avoid sounding scripted or overly formal
- Balance information sharing with entertainment

### Dialogue:"""


def get_dialogue_prompt_without_context(instruction: str, previous_conversation: dict = None) -> str:
    """
    Generate prompt for dialogue generation without RAG context
    
    Args:
        instruction: User's instruction for dialogue generation
        previous_conversation: Optional previous conversation memory
    
    Returns:
        Formatted prompt string
    """
    previous_context = ""
    if previous_conversation:
        previous_context = f"""\n### Previous Topic (For Reference Only):
{previous_conversation.get('instruction', 'N/A')}

Note: Consider the previous topic ONLY if the current instruction is a follow-up, continuation, or modification request. Otherwise, treat this as a new, independent conversation.\n"""
    
    return f"""Create an engaging podcast-style conversation between two hosts (Person A and Person B) based on the following topic.
{previous_context}
### Episode Topic:
{instruction}

### Podcast Style Guidelines:
- Format each line as "A: [text]" or "B: [text]"
- **IMPORTANT: Generate exactly 5 rounds of dialogue (10 lines total: A speaks, B responds, repeat 5 times)**
- Make it sound like a REAL podcast recording:
  * Use natural speech patterns with filler words (you know, I mean, like, um, right)
  * Include reactions and responses ("Oh really?", "That's fascinating!", "Exactly!", "Wait, what?")
  * Show genuine enthusiasm and curiosity
  * Add personal touches and relatable examples
  * Let hosts interrupt, agree, or build on each other's points naturally
  * Use contractions (don't, it's, we're) for casual tone
  * Don't need the opening remark
- Keep it conversational and engaging
- Add some humor or personality where appropriate
- If continuing a previous conversation, for example, the instruction says "No, I want to know more about...", generate responses with first sentences like "Well, speaking of that...", "Actually, that reminds me...", or "Building on what you just said..."
- Avoid sounding scripted or overly formal

### Dialogue:"""


DIALOGUE_SYSTEM_PROMPT = "You are a podcast script writer who creates engaging, natural conversations between two hosts. The dialogue should feel spontaneous, interactive, and entertaining, like a real podcast recording."


def get_dialogue_prompt_with_web_search(context: str, instruction: str, difficulty: str, web_search_context: str, previous_conversation: dict = None) -> str:
    """
    Generate prompt for dialogue generation with both RAG context and web search context
    
    Args:
        context: Retrieved context from RAG
        instruction: User's instruction for dialogue generation
        difficulty: Difficulty level (easy, medium, hard, professional)
        web_search_context: Additional context from web search
        previous_conversation: Optional previous conversation memory
    
    Returns:
        Formatted prompt string
    """
    previous_context = ""
    if previous_conversation:
        previous_context = f"""\n### Previous Topic (For Reference Only):
{previous_conversation.get('instruction', 'N/A')}

Note: Consider the previous topic ONLY if the current instruction is a follow-up, continuation, or modification request. Otherwise, treat this as a new, independent conversation.\n"""
    
    return f"""Create an engaging podcast-style conversation between two hosts (Person A and Person B) using information from both our knowledge base and latest web research.
{previous_context}
### Knowledge Base:
{context}

### Latest Web Research:
{web_search_context}

### Episode Topic/Instruction:
{instruction}

Difficulty Level: {difficulty}

### Podcast Style Guidelines:
- Format each line as "A: [text]" or "B: [text]"
- **IMPORTANT: Generate exactly 5 rounds of dialogue (10 lines total: A speaks, B responds, repeat 5 times)**
- Make it sound like a REAL podcast recording:
  * Use natural speech patterns with filler words (you know, I mean, like, um, right)
  * Include reactions and responses ("Oh really?", "That's fascinating!", "Exactly!", "Wait, what?")
  * Show genuine enthusiasm and curiosity
  * Add personal touches and relatable examples
  * Let hosts interrupt, agree, or build on each other's points naturally
  * Use contractions (don't, it's, we're) for casual tone
  * Don't need the opening remark
- Seamlessly blend information from both sources
- Mention when sharing recent updates or discoveries from web research
- Keep it conversational but informative
- Add some humor or personality where appropriate
- If continuing a previous conversation, for example, the instruction says "No, I want to know more about...", generate responses with first sentences like "Well, speaking of that...", "Actually, that reminds me...", or "Building on what you just said..."
- Avoid sounding scripted or overly formal
- Balance information sharing with entertainment

### Dialogue:"""