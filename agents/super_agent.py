def super_agent_router(state):
    user_msg = state["messages"][-1].content.lower()
    if "find" in user_msg or "search" in user_msg or "tell me about" in user_msg:
        return {"next": "rag_agent"}
    else:
        return {"next": "chat_agent"}

