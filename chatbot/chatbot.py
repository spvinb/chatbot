
from agents.ReActAgent import invoke_agent

def interactive_chat():
    print("\n\nWellcom to the IMD chatbot!! Ask you question or type 'exit' to stop.")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "stop"]:
            print ("ThankYou!!! for using IMD chatbot")
            break

        response = invoke_agent(query)
        print(response['output'])

if __name__ == "__main__":
    interactive_chat()
