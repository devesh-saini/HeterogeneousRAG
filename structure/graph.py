from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from .qwen import node_qwen

graph = StateGraph(MessagesState)
graph.add_node(node_qwen)
graph.add_edge(START, node_qwen)
graph.add_edge(node_qwen, END)
graph.compile()

if __name__ == "__main__":
    query = input(">>> ")
    graph.invoke(HumanMessage(content=query))

