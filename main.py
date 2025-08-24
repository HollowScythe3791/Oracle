from langgraph.graph import StateGraph, MessagesState, START, END
from agents.super_agent import super_agent_router
from agents.chat_agent import chat_agent_node
from agents.rag_agent import rag_agent_node

builder = StateGraph(MessagesState)
builder.add_node("super_agent", super_agent_router)
builder.add_node("chat_agent", chat_agent_node)
builder.add_node("rag_agent", rag_agent_node)
builder.add_edge(START, "super_agent")
builder.add_conditional_edges(
    "super_agent",
    lambda state: state["next"]
)
builder.add_edge("chat_agent", END)
builder.add_edge("rag_agent", END)

graph = builder.compile()


# Chat loop (for CLI demo)
from langchain_core.messages import HumanMessage
print("Welcome to the Modular LangGraph Super Agent! Type 'exit' to quit.")
messages = []
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break
    messages.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": messages})
    # Print only the latest assistant reply
    from langchain_core.messages import AIMessage
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print("Assistant:", msg.content)
    messages = result["messages"]
