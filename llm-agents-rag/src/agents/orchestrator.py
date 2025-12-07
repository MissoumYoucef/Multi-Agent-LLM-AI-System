from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .chatbot import ChatbotAgent
from .solver import SolverAgent
from .analyzer import AnalyzerAgent
from ..rag.retriever import HybridRetriever

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str
    problem: str
    solution: str
    analysis: str

class Orchestrator:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.chatbot = ChatbotAgent()
        self.solver = SolverAgent()
        self.analyzer = AnalyzerAgent()
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("solve", self.solve_node)
        workflow.add_node("analyze", self.analyze_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "solve")
        workflow.add_edge("solve", "analyze")
        workflow.add_edge("analyze", END)

        return workflow.compile()

    def retrieve_node(self, state: AgentState):
        query = state["messages"][-1].content
        print(f"--- Retrieving context for: {query} ---")
        docs = self.retriever.retrieve(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"context": context, "problem": query}

    def solve_node(self, state: AgentState):
        print("--- Solving ---")
        problem = state["problem"]
        context = state["context"]
        solution = self.solver.invoke(problem, context)
        return {"solution": solution, "messages": [AIMessage(content=solution)]}

    def analyze_node(self, state: AgentState):
        print("--- Analyzing ---")
        problem = state["problem"]
        solution = state["solution"]
        context = state["context"]
        analysis = self.analyzer.invoke(problem, solution, context)
        return {"analysis": analysis}

    def run(self, query: str):
        inputs = {"messages": [HumanMessage(content=query)]}
        return self.workflow.invoke(inputs)
