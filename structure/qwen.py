from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama

qwen = ChatOllama(model="qwen2.5:latest")


def node_qwen(state: MessagesState):
    response = qwen.invoke(AIMessage(content=state["messages"]))
    return response
