from langchain_core.messages import AIMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Temporary local storage
documents = SimpleDirectoryReader("data").load_data()
# Build the vector index
index = VectorStoreIndex.from_documents(documents)
# Create a query engine
query_engine = index.as_query_engine()

def rag_agent_node(state):
    user_query = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else str(state["messages"][-1])
    response = query_engine.query(user_query)
    return {"messages": [AIMessage(content=str(response))]}

