"""
Basic LangGraph Chatbot
- Takes the state, and input
- Adds a system prompt
- Sends it to an llm
- Returns the response
"""
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from dotenv import load_dotenv
load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-4o")

# System instruction for a helpful, friendly assistant
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful, friendly assistant. Answer clearly and concisely."
    ),
}

# Node: LLM chat logic
def llm_chat_node(state: MessagesState):
    # Prepend system prompt if not already present
    if not state["messages"]: 
        messages = [SYSTEM_PROMPT] + state["messages"]
    else:
        messages = state["messages"]
    return {"messages": [llm.invoke(messages)]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("llm_chat", llm_chat_node)
builder.add_edge(START, "llm_chat")
builder.add_edge("llm_chat", END)

# Compile graph
graph = builder.compile()

# --- Chat loop ---
print("Welcome to the LangGraph Chatbot! Type 'exit' to quit.")
messages = []

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break
    messages.append({"role": "user", "content": user_input})
    # Run the conversation turn through the graph
    result = graph.invoke({"messages": messages})
    # The assistant's reply will be the last message
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print("Assistant:", msg.content)
    # Update chat history with the latest messages
    messages = result["messages"]

print(messages)
