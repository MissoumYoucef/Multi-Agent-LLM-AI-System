"""
Enhanced Orchestrator module.

Orchestrates multiple agents with guardrails, ReAct reasoning, and self-reflection.
"""
import logging
from typing import TypedDict, Annotated, Sequence, Optional, Literal
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .chatbot import ChatbotAgent
from .solver import SolverAgent
from .analyzer import AnalyzerAgent
from .guardrails import InputGuardrail, OutputGuardrail, GuardrailStatus
from .reflective_agent import ReflectiveAgent
from .react_agent import SimpleReActAgent
from ..rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State shared between agents in the workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str
    problem: str
    solution: str
    analysis: str
    # Guardrail and reflection fields
    guardrail_status: str
    guardrail_violations: list
    reflection_score: float
    reflection_history: list
    reasoning_trace: str
    # Memory and session fields
    session_id: str
    conversation_history: str
    memory_summary: str
    # Error handling fields
    error_count: int
    last_error: str


# class Orchestrator:
#     """
#     Original orchestrator with basic workflow.
    
#     Maintained for backward compatibility.
#     """
    
#     def __init__(self, retriever: HybridRetriever):
#         self.retriever = retriever
#         self.chatbot = ChatbotAgent()
#         self.solver = SolverAgent()
#         self.analyzer = AnalyzerAgent()
#         self.workflow = self._create_workflow()

#     def _create_workflow(self):
#         workflow = StateGraph(AgentState)

#         workflow.add_node("retrieve", self.retrieve_node)
#         workflow.add_node("solve", self.solve_node)
#         workflow.add_node("analyze", self.analyze_node)

#         workflow.set_entry_point("retrieve")
#         workflow.add_edge("retrieve", "solve")
#         workflow.add_edge("solve", "analyze")
#         workflow.add_edge("analyze", END)

#         return workflow.compile()

#     def retrieve_node(self, state: AgentState):
#         query = state["messages"][-1].content
#         print(f"--- Retrieving context for: {query} ---")
#         docs = self.retriever.retrieve(query)
#         context = "\n\n".join([doc.page_content for doc in docs])
#         return {"context": context, "problem": query}

#     def solve_node(self, state: AgentState):
#         print("--- Solving ---")
#         problem = state["problem"]
#         context = state["context"]
#         solution = self.solver.invoke(problem, context)
#         return {"solution": solution, "messages": [AIMessage(content=solution)]}

#     def analyze_node(self, state: AgentState):
#         print("--- Analyzing ---")
#         problem = state["problem"]
#         solution = state["solution"]
#         context = state["context"]
#         analysis = self.analyzer.invoke(problem, solution, context)
#         return {"analysis": analysis}

#     def run(self, query: str):
#         inputs = {"messages": [HumanMessage(content=query)]}
#         return self.workflow.invoke(inputs)


class Orchestrator:
    """
    Enhanced orchestrator with guardrails, ReAct, and self-reflection.
    
    Workflow: Input Guardrail → [Optional ReAct] → Retrieve → Solve → 
              Analyze → Self-Reflect → Output Guardrail
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        enable_guardrails: bool = True,
        enable_reflection: bool = True,
        enable_react: bool = False,  # Off by default for faster processing
        reflection_threshold: float = 0.7
    ):
        """
        Initialize enhanced orchestrator.
        
        Args:
            retriever: The hybrid retriever for RAG.
            enable_guardrails: Whether to enable input/output guardrails.
            enable_reflection: Whether to enable self-reflection.
            enable_react: Whether to enable ReAct reasoning.
            reflection_threshold: Quality threshold for reflection.
        """
        self.retriever = retriever
        self.enable_guardrails = enable_guardrails
        self.enable_reflection = enable_reflection
        self.enable_react = enable_react
        
        # Initialize agents
        self.chatbot = ChatbotAgent()
        self.solver = SolverAgent()
        self.analyzer = AnalyzerAgent()
        
        # Initialize new pattern components
        if enable_guardrails:
            self.input_guardrail = InputGuardrail()
            self.output_guardrail = OutputGuardrail()
        
        if enable_reflection:
            self.reflective_agent = ReflectiveAgent(
                quality_threshold=reflection_threshold
            )
        
        if enable_react:
            self.react_agent = SimpleReActAgent()
        
        self.workflow = self._create_enhanced_workflow()
        logger.info(f"EnhancedOrchestrator initialized: guardrails={enable_guardrails}, "
                   f"reflection={enable_reflection}, react={enable_react}")

    def _create_enhanced_workflow(self):
        """Create the enhanced workflow with conditional nodes."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("input_guard", self.input_guard_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("reason", self.reason_node)
        workflow.add_node("solve", self.solve_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("reflect", self.reflect_node)
        workflow.add_node("output_guard", self.output_guard_node)

        # Set entry point
        workflow.set_entry_point("input_guard")
        
        # Add conditional edge from input guard
        workflow.add_conditional_edges(
            "input_guard",
            self._route_after_input_guard,
            {
                "blocked": END,
                "continue": "retrieve"
            }
        )
        
        # Main flow
        workflow.add_edge("retrieve", "reason" if self.enable_react else "solve")
        if self.enable_react:
            workflow.add_edge("reason", "solve")
        workflow.add_edge("solve", "analyze")
        workflow.add_edge("analyze", "reflect" if self.enable_reflection else "output_guard")
        if self.enable_reflection:
            workflow.add_edge("reflect", "output_guard")
        workflow.add_edge("output_guard", END)

        return workflow.compile()

    def _route_after_input_guard(self, state: AgentState) -> Literal["blocked", "continue"]:
        """Route based on input guardrail result."""
        if state.get("guardrail_status") == "fail":
            return "blocked"
        return "continue"

    def input_guard_node(self, state: AgentState):
        """Apply input guardrails."""
        query = state["messages"][-1].content
        print("--- Input Guardrail Check ---")
        
        if not self.enable_guardrails:
            return {
                "guardrail_status": "skip",
                "guardrail_violations": [],
                "problem": query
            }
        
        result = self.input_guardrail.validate(query)
        
        if result.failed:
            logger.warning(f"Input blocked: {result.violations}")
            return {
                "guardrail_status": "fail",
                "guardrail_violations": result.violations,
                "problem": query,
                "solution": f"Request blocked: {', '.join(result.violations)}",
                "analysis": "Input validation failed."
            }
        
        return {
            "guardrail_status": result.status.value,
            "guardrail_violations": result.violations,
            "problem": result.sanitized_input or query
        }

    def retrieve_node(self, state: AgentState):
        """Retrieve relevant context."""
        query = state["problem"]
        print(f"--- Retrieving context for: {query} ---")
        docs = self.retriever.retrieve(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"context": context}

    def reason_node(self, state: AgentState):
        """Apply ReAct reasoning if enabled."""
        print("--- ReAct Reasoning ---")
        
        if not self.enable_react:
            return {"reasoning_trace": "ReAct disabled"}
        
        problem = state["problem"]
        context = state["context"]
        
        reasoning = self.react_agent.invoke(problem, context)
        return {"reasoning_trace": reasoning}

    def solve_node(self, state: AgentState):
        """Generate solution."""
        print("--- Solving ---")
        problem = state["problem"]
        context = state["context"]
        
        # Include reasoning trace if available
        if self.enable_react and state.get("reasoning_trace"):
            context = f"{context}\n\nReasoning: {state['reasoning_trace']}"
        
        solution = self.solver.invoke(problem, context)
        return {"solution": solution, "messages": [AIMessage(content=solution)]}

    def analyze_node(self, state: AgentState):
        """Analyze the solution."""
        print("--- Analyzing ---")
        problem = state["problem"]
        solution = state["solution"]
        context = state["context"]
        analysis = self.analyzer.invoke(problem, solution, context)
        return {"analysis": analysis}

    def reflect_node(self, state: AgentState):
        """Apply self-reflection if enabled."""
        print("--- Self-Reflection ---")
        
        if not self.enable_reflection:
            return {
                "reflection_score": 1.0,
                "reflection_history": []
            }
        
        problem = state["problem"]
        solution = state["solution"]
        context = state["context"]
        
        result = self.reflective_agent.reflect_and_improve(
            question=problem,
            response=solution,
            context=context
        )
        
        # Update solution if improved
        if result["improvements_made"] > 0:
            print(f"--- Solution improved ({result['improvements_made']} iterations) ---")
        
        return {
            "solution": result["final_response"],
            "reflection_score": result["final_score"],
            "reflection_history": result["reflection_history"]
        }

    def output_guard_node(self, state: AgentState):
        """Apply output guardrails."""
        print("--- Output Guardrail Check ---")
        
        if not self.enable_guardrails:
            return {}
        
        result = self.output_guardrail.validate(
            state["solution"],
            question=state["problem"]
        )
        
        if result.failed:
            logger.warning(f"Output sanitized: {result.violations}")
            return {
                "solution": result.sanitized_input,
                "guardrail_violations": state.get("guardrail_violations", []) + result.violations
            }
        
        return {}

    def run(self, query: str) -> dict:
        """
        Run the enhanced workflow.
        
        Args:
            query: The user query.
            
        Returns:
            Dictionary with solution, analysis, and metadata.
        """
        inputs = {
            "messages": [HumanMessage(content=query)],
            "guardrail_status": "",
            "guardrail_violations": [],
            "reflection_score": 0.0,
            "reflection_history": [],
            "reasoning_trace": ""
        }
        return self.workflow.invoke(inputs)
