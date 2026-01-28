from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class GraphState(TypedDict):
    query: str
    context: List[str]
    sources: List[dict]
    answer: str
    evaluation: str

def build_graph(input_filter, retriever, answer_agent, evaluation_agent, vectorstore):
    graph = StateGraph(GraphState)

    graph.add_node("input_filter", input_filter)
    graph.add_node("retriever", lambda s: retriever(s, vectorstore))
    graph.add_node("answer", answer_agent)
    graph.add_node("evaluation", evaluation_agent)

    graph.set_entry_point("input_filter")
    graph.add_edge("input_filter", "retriever")
    graph.add_edge("retriever", "answer")
    graph.add_edge("answer", "evaluation")
    graph.add_edge("evaluation", END)

    return graph.compile()