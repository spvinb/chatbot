import os
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Set up API key and environment variables
os.environ["OPENAI_API_KEY"] = "your_openai_key"

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4")

@tool("advisory_tool", return_direct=True)
def advisory_tool(question: str) -> str:
    """
    Call this tool to respond to all types of advisory-related queries.
    - Handles questions involving clothing, travel, event planning, outdoor activities, health & safety, and more.
    - Leverages few-shot prompting to provide specific, context-aware recommendations.
    """

    advisory_template = """You are an assistant specializing in providing advisory recommendations based on user queries. Your goal is to provide clear, practical, and contextually relevant advice.

    Few-shot examples:
    1. Input: "Next week, I'm heading to Delhi. What should I carry with me?"
       Output: "Delhi next week is expected to be cold. Carry warm clothing such as jackets, sweaters, and a scarf. Also, pack moisturizer to protect your skin from dryness."

    2. Input: "What precautions should I take while traveling to South Bombay tomorrow?"
       Output: "South Bombay tomorrow may experience mild rainfall. Carry an umbrella, wear waterproof footwear, and avoid busy areas during peak hours to stay safe."

    3. Input: "Can I host an outdoor party this weekend in Bangalore?"
       Output: "Bangalore may have scattered showers this weekend. Consider setting up a tent or choosing a venue with a covered area to avoid disruptions."

    4. Input: "I'm planning a hiking trip near Manali. Is it a good idea?"
       Output: "Hiking near Manali is generally safe if there's no heavy snowfall. Check for weather updates, wear sturdy hiking boots, and carry essentials like water and a first-aid kit."

    Instructions:
    - Analyze the query to identify the context (e.g., location, time, type of activity).
    - Provide relevant recommendations or precautions based on the scenario.
    - If specific weather or environmental details are unavailable, recommend retrieving data for more accurate advice.

    Input Query: {question}
    """
    prompt = PromptTemplate.from_template(advisory_template)

    # Chain the prompt with the LLM
    chain = prompt | llm
    response = chain.invoke(question)
    return response
