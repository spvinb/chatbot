import spacy
from datetime import datetime, timedelta
from dateutil import parser
import re
from langchain.agents import tool

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

@tool("resolve_relative_date_tool", return_direct=True)
def resolve_relative_date_tool(question: str) -> str:
    """
    Call this tool to resolve relative date and time references in user queries to absolute dates.
    """
    
    today = datetime.now()

    # Helper function to resolve a single date phrase using dateutil
    def resolve_phrase(phrase: str):
        try:
            resolved_date = parser.parse(phrase, default=today)
            return resolved_date.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            return None

    # Parse the question with spaCy
    doc = nlp(question)
    resolved_question = question

    # Custom regular expressions for certain common date-related expressions
    def resolve_custom_phrases(question: str, today: datetime) -> str:
        # Patterns for common relative date expressions
        patterns = {
            r"next week": 7,  # "next week" means 7 days from today
            r"in (\d+) days": lambda match: int(match.group(1)),  # Extracts "in X days"
            r"(\d+) days later": lambda match: int(match.group(1)),  # Matches "X days later"
            r"tomorrow": 1,  # "tomorrow" means 1 day from today
            r"yesterday": -1,  # "yesterday" means 1 day before today
            r"last Monday": -7,  # Assumes last Monday is exactly 7 days ago
            r"next Monday": 7,  # Next Monday (7 days from today)
            r"this Friday": 5,  # Adjust as per current weekday
        }
        
        for pattern, value in patterns.items():
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                days = value(match) if callable(value) else value
                resolved_date = (today + timedelta(days=days)).strftime("%Y-%m-%d")
                question = re.sub(pattern, resolved_date, question, flags=re.IGNORECASE)
        
        return question

    # Apply custom temporal phrase resolution
    resolved_question = resolve_custom_phrases(resolved_question, today)

    # Now, apply spaCy's NER for more complex temporal entities
    for ent in doc.ents:
        if ent.label_ == "DATE":
            resolved_date = resolve_phrase(ent.text)
            if resolved_date:
                resolved_question = resolved_question.replace(ent.text, resolved_date)

    return resolved_question


# # Test the function
# if __name__ == "__main__":
#     query = input("Enter the query:\n")
#     print(resolve_relative_date_tool(query))