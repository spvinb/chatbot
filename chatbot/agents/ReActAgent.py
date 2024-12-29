# Langchain Imports

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from pydantic import BaseModel

# Import tools
from tools.InteractionhandlerTool.interactionhandler import interactionhandler
from tools.AdvisoryTool.advisory_tool import advisory_tool
from tools.RagTool.rag_tool import weather_data_retrieval
from tools.ResolvequestionTool.resolve_question_tool import resolve_relative_date_tool
# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = api_key


llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = [interactionhandler, advisory_tool, weather_data_retrieval, resolve_relative_date_tool]

def invoke_agent(question):

    agent_template = hub.pull("hwchase17/react")

    # agent_template = """
    
    # Answer the following questions as best you can. You have access to the following tools:

    # {tools}

    # Use the following format:

    # Question: the input question you must answer
    # Thought: you should always think about what to do
    # Action: the action to take, should be one of [{tool_names}]
    # Action Input: the input to the action
    # Observation: the result of the action
    # ... (this Thought/Action/Action Input/Observation can repeat N times)
    # Thought: I now know the final answer
    # Final Answer: the final answer to the original input question

    # Begin!

    # Question: {input}
    # Thought:{agent_scratchpad}
    # """

    agent = create_react_agent(
        llm=llm,
        prompt=agent_template,
        tools=tools,
    )

    agent_executeor = AgentExecutor(
        agent=agent,
        verbose=True,
        tools=tools,
    )

    response = agent_executeor.invoke({"input": question})

    return response