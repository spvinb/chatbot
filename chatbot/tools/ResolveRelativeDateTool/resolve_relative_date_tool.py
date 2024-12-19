from datetime import datetime
from dateutil import parser
from langchain.agents import tool
import spacy

# Load spaCy NLP model
# Download the model using: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

@tool("resolve_relative_date_tool", return_direct=True)
def resolve_relative_date_tool(question: str) -> str:
    """
    Call this tool to resolve relative date and time references in user queries to absolute dates.
    
    - Uses spaCy's named entity recognition (NER) to identify temporal phrases like 'next week', 'in 3 days', 
      'two weeks from now', etc.
    - Resolves identified temporal expressions into exact dates using the dateutil library.
    - Replaces the temporal references in the question with their resolved absolute dates.
    
    **Examples**:
    
    Input Query: "What will the weather be like next week?"
    Resolved Query: "What will the weather be like from 2024-12-25 to 2024-12-31?"
    
    Input Query: "Tell me the weather two days later."
    Resolved Query: "Tell me the weather on 2024-12-21."
    
    Input Query: "What precautions should I take for my trip in 3 days?"
    Resolved Query: "What precautions should I take for my trip on 2024-12-22?"
    """

    # Current date
    today = datetime.now()

    # Helper function to resolve a single date phrase
    def resolve_phrase(phrase: str):
        try:
            # Use dateutil parser to resolve phrases
            resolved_date = parser.parse(phrase, default=today)
            return resolved_date.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            return None

    # Parse the question using spaCy
    doc = nlp(question)
    resolved_question = question

    
    # Iterate over the recognized date entities
    for ent in doc.ents:
        if ent.label_ == "DATE":
            # Attempt to resolve the date phrase
            resolved_date = resolve_phrase(ent.text)
            if resolved_date:
                resolved_question = resolved_question.replace(ent.text, resolved_date)

    return resolved_question
