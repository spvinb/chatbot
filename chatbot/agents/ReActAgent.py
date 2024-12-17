# Langchain Imports

import os
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub

# Import tools
from tools.InteractionhandlerTool import interactionhandler

os.environ["OPENAI_API_KEY"] = "your_openai_key"

llm = ChatOpenAI(model="gpt-4")
tools = [interactionhandler]

def invoke_agent():

    agent_template = hub.pull("hwchase17/react")
    # agent_template = """
    # """
    prompt = PromptTemplate.from_template(agent_template)

    agent = create_react_agent(
        llm=llm,
        prompt=prompt,
        tools=tools,
    )

    agent_executeor = AgentExecutor(
        agent=agent,
        verbose=True,
        tools=tools,
    )

    return agent_executeor