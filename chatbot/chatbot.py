import os
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub

# Import tools
from tools.InteractionhandlerTool import interactionhandler
from tools.RagTool import rag_tool
from ResolveRelativeDateTool import resolve_relative_date_tool
from AdvisoryTool import advisory_tool

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your_openai_key"  # Replace with your actual OpenAI API key

# Initialize the LLM (GPT-4 model from OpenAI)
llm = ChatOpenAI(model="gpt-4")

# List of tools available for the agent
tools = [interactionhandler, rag_tool, resolve_relative_date_tool, advisory_tool]

def invoke_agent():
    """
    Initializes and returns the agent executor using the provided tools and agent template.
    """
    # Fetch the React agent template from LangChain Hub
    agent_template = hub.pull("hwchase17/react")

    # Create the prompt using the fetched template
    prompt = PromptTemplate.from_template(agent_template)

    # Create the React agent with LLM, prompt, and tools
    agent = create_react_agent(
        llm=llm,
        prompt=prompt,
        tools=tools,
    )

    # Initialize the AgentExecutor which will execute the agent with the provided tools
    agent_executor = AgentExecutor(
        agent=agent,
        verbose=True,
        tools=tools,
    )

    return agent_executor

def start_chatbot():
    """
    Starts the chatbot, processes user input, and provides responses iteratively.
    """
    # Initialize the agent executor
    agent_executor = invoke_agent()

    print("Welcome to the Chatbot! Type 'exit' to quit.")
    while True:
        # Accept input from the user
        user_input = input("Input Question: ")

        # If user types 'exit', end the chat
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Parse the question and send it to the agent for processing
        result = agent_executor.invoke({"input": user_input})

        # Display the result from the agent
        print("Agent Response:", result)

if __name__ == "__main__":
    # Start the chatbot interactive Q&A loop
    start_chatbot()
