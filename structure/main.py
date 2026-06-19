from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama


qwen_model = ChatOllama(model="qwen2.5-coder:7b")


def qwen_node(state: MessagesState):
    response = qwen_model.invoke(state["messages"])
    return {"messages": [AIMessage(response.content)]}


graph = StateGraph(MessagesState)
graph.add_node("call_qwen", qwen_node)
graph.add_edge(START, "call_qwen")
graph.add_edge("call_qwen", END)
compiled_graph = graph.compile()


if __name__ == "__main__":
    message = input(">>> ")
    result = compiled_graph.invoke({"messages": [HumanMessage(content=message)]})
    print(result["messages"][-1].content)
