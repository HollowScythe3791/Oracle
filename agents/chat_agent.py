from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

def chat_agent_node(state):
    last_user_msg = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])
    response = llm.invoke([{"role": "user", "content": last_user_msg}])
    return {"messages": [AIMessage(content=response.content)]}
