import os 
from langchain.agents import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


os.environ["OPENAI_API_KEY"] = "your_openai_key"

llm = ChatOpenAI(model="gpt-4")


@tool("interactionhandler", return_direct=True)
def interactionhandler(question: str) -> str:
    """
    Use this tool to respond to greeting.
    - Offer a warm response to common greetings such as "Good morning", "Hello", "Hi" etc.
    - Focus on delivering brief and appropriate replies without delving into additional information or context.
    """

    interaction_template = """"Greet user while subtly mentioning IMD weather broadcast when appropriate. Input:{question} """
    prompt = PromptTemplate.from_template(interaction_template)
    chain = prompt | llm 
    response = chain.invoke(question)
    return response