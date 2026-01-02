"""
Enhanced Orchestrator module.

Orchestrates multiple agents with guardrails, ReAct reasoning, self-reflection,
and integrated utilities for caching, cost control, memory, and evaluation.
"""
import logging
import time
from typing import TypedDict, Annotated, Sequence, Optional, Literal, Dict, Any, Protocol, List, TYPE_CHECKING
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

from .chatbot import ChatbotAgent
from .solver import SolverAgent
from .analyzer import AnalyzerAgent
from .guardrails import InputGuardrail, OutputGuardrail, GuardrailStatus
from .reflective_agent import ReflectiveAgent
from .react_agent import SimpleReActAgent

# Define a Protocol for retrievers to decouple from the actual HybridRetriever class
# This allows the orchestrator to work with any object that has a retrieve() method
class RetrieverProtocol(Protocol):
    """Protocol for retriever objects. Any object with a matching retrieve method will work."""
    def retrieve(self, query: str) -> List[Document]:
        ...

# Conditional import - only for type checking, not at runtime
# This prevents ModuleNotFoundError when running in the inference-service container
if TYPE_CHECKING:
    from ..rag.retriever import HybridRetriever

# Integrated utility imports
from ..utils.cache import ResponseCache
from ..utils.cost_controller import CostController, BudgetPeriod
from ..utils.token_manager import TokenManager
from ..utils.config import (
    CACHE_ENABLED, CACHE_TTL_SECONDS, DAILY_BUDGET_USD,
    COST_ALERT_THRESHOLD, MEMORY_BUFFER_SIZE, REDIS_URL
)
from ..memory.memory_manager import MemoryManager
from ..evaluation.continuous_eval import ContinuousEvaluator
from .error_handling import RetryHandler, CircuitBreaker, RetryConfig

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
    # Cache and cost control fields
    cache_hit: bool
    cost_estimate: float
    token_count: int
    quality_score: float


class Orchestrator:
    """
    Orchestrator with guardrails, ReAct, self-reflection, and integrated Cache and cost control.

    Workflow: Input Guardrail → [Cache Check] → [Optional ReAct] → Retrieve → Solve →
              Analyze → Self-Reflect → [Evaluate] → Output Guardrail → [Cache Store]

    Integrated utilities:
    - ResponseCache: Caches responses to avoid redundant LLM calls
    - CostController: Tracks and controls API costs
    - TokenManager: Counts tokens and truncates context when needed
    - MemoryManager: Session-based conversation memory
    - ContinuousEvaluator: Monitors response quality
    - RetryHandler/CircuitBreaker: Error resilience
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        enable_guardrails: bool = True,
        enable_reflection: bool = True,
        enable_react: bool = False,  # Off by default for faster processing
        reflection_threshold: float = 0.7,
        enable_caching: bool = True,
        enable_cost_control: bool = True,
        enable_memory: bool = True,
        enable_evaluation: bool = True,
        daily_budget: float = None,
        session_id: str = "default"
    ):
        """
        Initialize orchestrator.

        Args:
            retriever: The hybrid retriever for RAG.
            enable_guardrails: Whether to enable input/output guardrails.
            enable_reflection: Whether to enable self-reflection.
            enable_react: Whether to enable ReAct reasoning.
            reflection_threshold: Quality threshold for reflection.
            enable_caching: Whether to enable response caching.
            enable_cost_control: Whether to enable cost tracking/control.
            enable_memory: Whether to enable conversation memory.
            enable_evaluation: Whether to enable quality evaluation.
            daily_budget: Daily budget in USD (uses config default if None).
            session_id: Default session ID for memory.
        """
        self.retriever = retriever
        self.enable_guardrails = enable_guardrails
        self.enable_reflection = enable_reflection
        self.enable_react = enable_react
        self.session_id = session_id

        # Initialize agents
        self.chatbot = ChatbotAgent()
        self.solver = SolverAgent()
        self.analyzer = AnalyzerAgent()

        # Initialize pattern components
        if enable_guardrails:
            self.input_guardrail = InputGuardrail()
            self.output_guardrail = OutputGuardrail()

        if enable_reflection:
            self.reflective_agent = ReflectiveAgent(
                quality_threshold=reflection_threshold
            )

        if enable_react:
            self.react_agent = SimpleReActAgent()

        # Initialize integrated utilities
        self.enable_caching = enable_caching and CACHE_ENABLED
        self.enable_cost_control = enable_cost_control
        self.enable_memory = enable_memory
        self.enable_evaluation = enable_evaluation

        if self.enable_caching:
            self.cache = ResponseCache(
                max_size=1000,
                default_ttl=CACHE_TTL_SECONDS,
                enable_semantic=False,  # Simple exact-match caching
                redis_url=REDIS_URL
            )
            logger.info("Response caching enabled")

        if self.enable_cost_control:
            budget = daily_budget or DAILY_BUDGET_USD
            self.cost_controller = CostController(
                daily_budget=budget,
                alert_threshold=COST_ALERT_THRESHOLD,
                enable_hard_limit=False  # Soft limit by default
            )
            logger.info(f"Cost control enabled: ${budget}/day budget")

        if self.enable_memory:
            self.memory_manager = MemoryManager(
                buffer_size=MEMORY_BUFFER_SIZE,
                enable_summarization=False,  # Off for cost savings
                enable_persistence=True
            )
            logger.info("Conversation memory enabled")

        if self.enable_evaluation:
            self.evaluator = ContinuousEvaluator(
                quality_threshold=reflection_threshold,
                enable_shadow_eval=True
            )
            logger.info("Continuous evaluation enabled")

        # Token manager for context truncation
        self.token_manager = TokenManager()

        # Error handling with retry and circuit breaker
        self.retry_handler = RetryHandler(
            config=RetryConfig(max_retries=3, initial_delay=1.0)
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout=60.0
        )

        self.workflow = self._create_workflow()
        logger.info(f"Orchestrator initialized: guardrails={enable_guardrails}, "
                   f"reflection={enable_reflection}, react={enable_react}, "
                   f"caching={self.enable_caching}, cost_control={self.enable_cost_control}, "
                   f"memory={self.enable_memory}, evaluation={self.enable_evaluation}")

    def _create_workflow(self):
        """Create the workflow with conditional nodes."""
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
        session_id = state.get("session_id", self.session_id)
        print("--- Input Guardrail Check ---")

        # Check cache first
        if self.enable_caching:
            cached_response = self.cache.get(query)
            if cached_response:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return {
                    "guardrail_status": "skip",
                    "guardrail_violations": [],
                    "problem": query,
                    "solution": cached_response,
                    "analysis": "Retrieved from cache",
                    "cache_hit": True
                }

        # Check cost budget
        if self.enable_cost_control:
            can_proceed, message = self.cost_controller.check_budget()
            if not can_proceed:
                logger.warning(f"Budget exceeded: {message}")
                return {
                    "guardrail_status": "fail",
                    "guardrail_violations": [message],
                    "problem": query,
                    "solution": f"Request blocked: {message}",
                    "analysis": "Budget limit reached."
                }

        # Add to memory
        if self.enable_memory:
            self.memory_manager.add_user_message(query, session_id)

        if not self.enable_guardrails:
            return {
                "guardrail_status": "skip",
                "guardrail_violations": [],
                "problem": query,
                "cache_hit": False
            }

        result = self.input_guardrail.validate(query)

        if result.failed:
            logger.warning(f"Input blocked: {result.violations}")
            return {
                "guardrail_status": "fail",
                "guardrail_violations": result.violations,
                "problem": query,
                "solution": f"Request blocked: {', '.join(result.violations)}",
                "analysis": "Input validation failed.",
                "cache_hit": False
            }

        return {
            "guardrail_status": result.status.value,
            "guardrail_violations": result.violations,
            "problem": result.sanitized_input or query,
            "cache_hit": False
        }

    def retrieve_node(self, state: AgentState):
        """Retrieve relevant context."""
        # Skip if cache hit
        if state.get("cache_hit"):
            return {}

        query = state["problem"]
        print(f"--- Retrieving context for: {query} ---")

        # Use circuit breaker for retrieval
        try:
            docs = self.circuit_breaker.call(self.retriever.retrieve, query)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            docs = []

        context = "\n\n".join([doc.page_content for doc in docs])

        # Truncate context if too long
        token_count = self.token_manager.count_tokens(context)
        max_context_tokens = 4000  # Leave room for prompt and response
        if token_count > max_context_tokens:
            context = self.token_manager.truncate_to_limit(
                context,
                max_context_tokens,
                preserve_sentences=True
            )
            logger.info(f"Context truncated from {token_count} to {max_context_tokens} tokens")

        return {"context": context, "token_count": token_count}

    def reason_node(self, state: AgentState):
        """Apply ReAct reasoning if enabled."""
        # Skip if cache hit
        if state.get("cache_hit"):
            return {}

        print("--- ReAct Reasoning ---")

        if not self.enable_react:
            return {"reasoning_trace": "ReAct disabled"}

        problem = state["problem"]
        context = state["context"]

        # Use retry handler for reasoning
        result = self.retry_handler.execute(
            self.react_agent.invoke, problem, context
        )

        if result.success:
            return {"reasoning_trace": result.result}
        else:
            logger.warning(f"ReAct reasoning failed after {result.attempts} attempts")
            return {"reasoning_trace": "Reasoning failed"}

    def solve_node(self, state: AgentState):
        """Generate solution."""
        # Skip if cache hit
        if state.get("cache_hit"):
            return {}

        print("--- Solving ---")
        problem = state["problem"]
        context = state["context"]

        # Include reasoning trace if available
        if self.enable_react and state.get("reasoning_trace"):
            context = f"{context}\n\nReasoning: {state['reasoning_trace']}"

        # Include conversation history if memory enabled
        if self.enable_memory:
            session_id = state.get("session_id", self.session_id)
            memory_context = self.memory_manager.get_context(session_id)
            if memory_context:
                context = f"Conversation History:\n{memory_context}\n\n{context}"

        # Use retry handler for solving
        result = self.retry_handler.execute(
            self.solver.invoke, problem, context
        )

        if result.success:
            solution = result.result
        else:
            logger.error(f"Solver failed after {result.attempts} attempts")
            solution = "I apologize, but I encountered an error while processing your request."

        # Estimate and record cost
        if self.enable_cost_control:
            input_tokens = self.token_manager.count_tokens(f"{problem}\n{context}")
            output_tokens = self.token_manager.count_tokens(solution)
            cost = self.token_manager.estimate_cost(input_tokens, output_tokens)
            self.cost_controller.record_usage(
                model="gemini-pro",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            )

        return {"solution": solution, "messages": [AIMessage(content=solution)]}

    def analyze_node(self, state: AgentState):
        """Analyze the solution."""
        # Skip if cache hit
        if state.get("cache_hit"):
            return {}

        print("--- Analyzing ---")
        problem = state["problem"]
        solution = state["solution"]
        context = state["context"]
        analysis = self.analyzer.invoke(problem, solution, context)
        return {"analysis": analysis}

    def reflect_node(self, state: AgentState):
        """Apply self-reflection if enabled."""
        # Skip if cache hit
        if state.get("cache_hit"):
            return {}

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
        """Apply output guardrails and post-processing."""
        print("--- Output Guardrail Check ---")
        session_id = state.get("session_id", self.session_id)

        # Skip guardrails if cache hit (already validated)
        if state.get("cache_hit"):
            return {}

        # Evaluate response quality
        if self.enable_evaluation:
            eval_result = self.evaluator.evaluate_response(
                query=state["problem"],
                response=state["solution"],
                context=state.get("context", ""),
                session_id=session_id
            )
            quality_score = eval_result.quality_score
        else:
            quality_score = state.get("reflection_score", 1.0)

        # Store in cache
        if self.enable_caching and not state.get("cache_hit"):
            self.cache.set(
                query=state["problem"],
                response=state["solution"],
                metadata={"quality_score": quality_score}
            )

        # Store response in memory
        if self.enable_memory:
            self.memory_manager.add_assistant_message(state["solution"], session_id)

        if not self.enable_guardrails:
            return {"quality_score": quality_score}

        result = self.output_guardrail.validate(
            state["solution"],
            question=state["problem"]
        )

        if result.failed:
            logger.warning(f"Output sanitized: {result.violations}")
            return {
                "solution": result.sanitized_input,
                "guardrail_violations": state.get("guardrail_violations", []) + result.violations,
                "quality_score": quality_score
            }

        return {"quality_score": quality_score}

    def run(self, query: str, session_id: str = None) -> dict:
        """
        Run the enhanced workflow.

        Args:
            query: The user query.
            session_id: Optional session ID for memory (uses default if None).

        Returns:
            Dictionary with solution, analysis, and metadata.
        """
        inputs = {
            "messages": [HumanMessage(content=query)],
            "guardrail_status": "",
            "guardrail_violations": [],
            "reflection_score": 0.0,
            "reflection_history": [],
            "reasoning_trace": "",
            "session_id": session_id or self.session_id,
            "cache_hit": False,
            "quality_score": 0.0
        }
        return self.workflow.invoke(inputs)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all integrated utilities."""
        stats = {}

        if self.enable_caching:
            stats["cache"] = self.cache.get_stats()

        if self.enable_cost_control:
            stats["cost"] = {
                "daily_usage": self.cost_controller.get_usage(BudgetPeriod.DAILY),
                "budget_status": self.cost_controller.get_budget_status(BudgetPeriod.DAILY)
            }

        if self.enable_memory:
            stats["memory"] = {
                "sessions": self.memory_manager.list_sessions(),
                "session_stats": self.memory_manager.get_session_stats(self.session_id)
            }

        if self.enable_evaluation:
            stats["evaluation"] = self.evaluator.get_stats()

        return stats
