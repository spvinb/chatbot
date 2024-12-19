import os
from langchain.agents import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Set up API keys and environment variables
os.environ["OPENAI_API_KEY"] = "your_openai_key"

# Initialize the LLM and embedding model
llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

# Connect to ChromaDB
vectorstore = Chroma(collection_name="weather_data", embedding_function=embeddings)

# Set up the retrieval chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

@tool("weather_data_retrieval", return_direct=True)
def weather_data_retrieval(question: str) -> str:
    """
    Call this tool to retrieve weather-related data.
    - Handles queries requiring weather information, such as forecasts, conditions, and recommendations.
    - Retrieves relevant information from the ChromaDB vector database and formats it appropriately.
    """

    weather_template = """You are a weather assistant. Answer the user's query using the most relevant data retrieved from the vector database.
    Input: {question}"""
    prompt = PromptTemplate.from_template(weather_template)

    # Combine the input with retrieval chain
    chain = prompt | qa_chain
    response = chain.invoke(question)
    return response
