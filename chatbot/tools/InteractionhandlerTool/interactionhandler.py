import os
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI( model="gpt-3.5-turbo", temperature=0.5)

parser = StrOutputParser()

@tool("interactionhandler", return_direct=True)
def interactionhandler(question: str) -> str:
    """
    Use this tool to respond to greeting.
    - Offer a warm response to common greetings such as "Good morning", "Hello", "Hi" etc.
    - Focus on delivering brief and appropriate replies without delving into additional information or context.
    """

    interaction_template = """"Greet user while subtly mentioning IMD weather broadcast when appropriate. Input:{question} """
    prompt = PromptTemplate( template= interaction_template, input_variables=["question"])
    # prompt = PromptTemplate.from_template(interaction_template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(question)
    return parser.parse(response)