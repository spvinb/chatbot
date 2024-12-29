import os
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = api_key

parser = StrOutputParser()

def retrive_and_get_answer(question: str) -> str:
    # Initialize the LLM and embedding model
    llm = ChatOpenAI(model="gpt-4")
    embeddings = OpenAIEmbeddings()
    persist_directory = 'vector_db'

    vectorstore = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
    weather_template = """"
    Answer the question based only on the following context:
    {context}
     - -
    Answer the question based on the above context: {question}
    """
    prompt = PromptTemplate(input_variables = ["context", "question"], template = weather_template)


    retriever = vectorstore.as_retriever(search_type= "similarity",search_kwargs={"k": 10})
    # retriever_function = lambda question: retriever.get_relevant_documents(question) 
    # retrievel = RunnableParallel( {"context": retriever_function, "question": question} )
    

    retrievel = RunnableParallel(
        {"context": retriever , "question": RunnablePassthrough()}
    )

    qa_chain = retrievel | prompt | llm | StrOutputParser()
    answer = qa_chain.invoke(question)
    return answer


@tool("weather_data_retrieval", return_direct=True)
def weather_data_retrieval(question: str) -> str:
    """
    Use this tool to retrieve weather-related data.
    - Handles queries requiring weather information, such as forecasts, conditions, and recommendations.
    """

    response = retrive_and_get_answer(question)
    return response
