from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from langchain.agents import tool

@tool("resolve_question_tool", return_direct=True)
def resolve_question_tool(question: str) -> str:
    """
    Resolves relative date and time references (e.g., 'next week', 'tomorrow', 'in 3 days') into absolute dates.
    - Parses the user's question and replaces relative references with exact dates.
    - Supports day-based (e.g., 'next 2 days'), week-based (e.g., 'next week'), and month-based (e.g., 'next month') resolutions.
    """

    # Today's date
    today = datetime.now()

    # Helper function to calculate date ranges
    def resolve_date_reference(reference: str):
        if "today" in reference.lower():
            return today.strftime("%Y-%m-%d")
        elif "tomorrow" in reference.lower():
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "next week" in reference.lower():
            start = today + timedelta(days=(7 - today.weekday()))
            end = start + timedelta(days=6)
            return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        elif "next month" in reference.lower():
            start = (today + relativedelta(months=1)).replace(day=1)
            end = (start + relativedelta(months=1)) - timedelta(days=1)
            return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        elif "next" in reference.lower() and "days" in reference.lower():
            num_days = int(reference.lower().split("next")[1].split("days")[0].strip())
            start = today + timedelta(days=1)
            end = today + timedelta(days=num_days)
            return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        else:
            return None

    # Parse the question for date references
    resolved_question = question
    if "next" in question.lower() or "tomorrow" in question.lower() or "today" in question.lower():
        # Identify and resolve time references
        phrases = ["today", "tomorrow", "next week", "next month"]
        for phrase in phrases:
            if phrase in question.lower():
                resolved_date = resolve_date_reference(phrase)
                if resolved_date:
                    resolved_question = resolved_question.replace(phrase, resolved_date)
        # Handle "next X days"
        if "next" in question.lower() and "days" in question.lower():
            start_idx = question.lower().find("next")
            end_idx = question.lower().find("days") + len("days")
            phrase = question[start_idx:end_idx]
            resolved_date = resolve_date_reference(phrase)
            if resolved_date:
                resolved_question = resolved_question.replace(phrase, resolved_date)

    return resolved_question
