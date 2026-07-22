from __future__ import annotations

from langgraph.graph import END, StateGraph

from pipeline.nodes.reranker import reranker_node
from pipeline.nodes.retriever import retriever_node
from pipeline.nodes.rewriter import rewriter_node
from pipeline.nodes.synthesizer import synthesizer_node
from pipeline.nodes.verifier import verifier_node
from pipeline.state import RAGState


def route_after_verification(state: RAGState) -> str:
    if not state.get("verification_result", {}).get("verdict", False):
        if int(state.get("retry_count", 0)) < 2:
            return "rewriter"
    return "end"


def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("rewriter", rewriter_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("rewriter")
    graph.add_edge("rewriter", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "synthesizer")
    graph.add_edge("synthesizer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verification,
        {"rewriter": "rewriter", "end": END},
    )
    return graph.compile()
