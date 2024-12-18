from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4")

@tool("activity_tool", return_direct=True)
def activity_tool(question: str) -> str:
    """
    Use this tool to respond to queries implicitly requiring weather-based activity recommendations.
    - Handles questions related to clothing, travel, events, outdoor activities, and health & safety.
    - Identifies implicit weather dependencies and provides suggestions based on expected conditions.
    """

    activity_template = """You are an assistant specializing in activity planning based on weather conditions.
    Analyze the input query to identify implicit weather-related concerns and provide appropriate recommendations.

    Instructions:
    - If the query involves:
        - Clothing advice: Suggest attire based on weather conditions (e.g., "What should I wear in winter?").
        - Travel planning: Suggest travel-related tips or plans depending on the destination's weather.
        - Event planning: Provide insights into outdoor or indoor event suitability based on weather forecasts.
        - Outdoor activities: Advise on the feasibility of outdoor sports, picnics, or hikes based on conditions.
        - Health and safety: Recommend precautions like umbrellas, sunscreen, or warm clothing.
    - Include specific references to the weather if available or suggest retrieving data for precise planning.
    
    Input Query: {question}
    """
    prompt = PromptTemplate.from_template(activity_template)

    # Chain the prompt with the LLM
    chain = prompt | llm
    response = chain.invoke(question)
    return response
