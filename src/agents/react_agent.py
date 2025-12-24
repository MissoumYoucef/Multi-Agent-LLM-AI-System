"""
ReAct Agent module.

Implements the ReAct (Reason + Act) pattern for multi-step reasoning.
The agent explicitly thinks through problems before taking actions.
"""
import logging
from typing import List, Optional, Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.agents import AgentExecutor, create_react_agent
from langchain_classic.agents import create_react_agent
# from langchain_core.agents import AgentExecutor
from langchain_classic.agents import AgentExecutor

from langchain_core.tools import BaseTool

from .tools import get_tools, calculate, summarize, format_as_list, extract_keywords
from ..utils.config import GOOGLE_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


# Custom ReAct prompt template
REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class ReActAgent:
    """
    ReAct (Reason + Act) Agent.
    
    Implements explicit reasoning before each action, following the
    Thought -> Action -> Observation loop pattern.
    """
    
    def __init__(
        self,
        model: str = None,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 5,
        verbose: bool = True
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            model: Optional model name override.
            tools: Optional list of tools. Defaults to standard tools.
            max_iterations: Maximum reasoning iterations before stopping.
            verbose: Whether to log detailed reasoning steps.
        """
        self.model_name = model or LLM_MODEL
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3  # Lower temperature for more focused reasoning
        )
        
        # Set up tools (exclude search_documents as it needs retriever injection)
        self.tools = tools or [calculate, summarize, format_as_list, extract_keywords]
        
        # Create the ReAct prompt
        self.prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
        
        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create the executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        logger.info(f"ReActAgent initialized with {len(self.tools)} tools")

    def invoke(self, question: str) -> Dict[str, Any]:
        """
        Process a question using ReAct reasoning.
        
        Args:
            question: The question to answer.
            
        Returns:
            Dictionary with 'output', 'intermediate_steps', and 'reasoning'.
        """
        if not question or not question.strip():
            logger.warning("Empty question received")
            return {
                "output": "Please provide a valid question.",
                "intermediate_steps": [],
                "reasoning": []
            }
        
        try:
            # Run the agent
            result = self.executor.invoke({"input": question})
            
            # Extract reasoning steps
            reasoning = self._extract_reasoning(result.get("intermediate_steps", []))
            
            logger.info(f"ReAct completed with {len(reasoning)} reasoning steps")
            
            return {
                "output": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"ReAct agent error: {e}")
            return {
                "output": f"Error during reasoning: {str(e)}",
                "intermediate_steps": [],
                "reasoning": []
            }

    def _extract_reasoning(self, steps: List) -> List[Dict[str, str]]:
        """
        Extract readable reasoning from intermediate steps.
        
        Args:
            steps: List of (AgentAction, observation) tuples.
            
        Returns:
            List of reasoning step dictionaries.
        """
        reasoning = []
        for i, (action, observation) in enumerate(steps):
            step_info = {
                "step": i + 1,
                "thought": getattr(action, 'log', str(action)),
                "action": action.tool if hasattr(action, 'tool') else str(action),
                "action_input": str(action.tool_input) if hasattr(action, 'tool_input') else "",
                "observation": str(observation)[:500]  # Truncate long observations
            }
            reasoning.append(step_info)
        return reasoning

    def get_reasoning_trace(self, result: Dict[str, Any]) -> str:
        """
        Format the reasoning trace as a readable string.
        
        Args:
            result: Result from invoke().
            
        Returns:
            Formatted reasoning trace.
        """
        reasoning = result.get("reasoning", [])
        if not reasoning:
            return "No intermediate reasoning steps."
        
        trace_lines = ["=== Reasoning Trace ==="]
        for step in reasoning:
            trace_lines.append(f"\n--- Step {step['step']} ---")
            trace_lines.append(f"Thought: {step['thought']}")
            trace_lines.append(f"Action: {step['action']}")
            trace_lines.append(f"Action Input: {step['action_input']}")
            trace_lines.append(f"Observation: {step['observation']}")
        
        trace_lines.append("\n=== End Trace ===")
        return '\n'.join(trace_lines)


class SimpleReActAgent:
    """
    A simpler ReAct implementation without AgentExecutor.
    
    Useful for more controlled reasoning without full agent framework.
    """
    
    def __init__(self, model: str = None, max_steps: int = 3):
        """
        Initialize simple ReAct agent.
        
        Args:
            model: Optional model name override.
            max_steps: Maximum reasoning steps.
        """
        self.model_name = model or LLM_MODEL
        self.max_steps = max_steps
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=GOOGLE_API_KEY
        )
        
        self.prompt = PromptTemplate(
            template="""You are a reasoning assistant. Think step by step.

                Question: {question}
                Context: {context}

                Let's approach this step by step:

                Step 1 - Understand: What is the question asking?
                Step 2 - Analyze: What information from the context is relevant?
                Step 3 - Reason: How can we combine the information to answer?
                Step 4 - Conclude: What is the final answer?

                Your reasoning:""",
                input_variables=["question", "context"]
            )
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info(f"SimpleReActAgent initialized")

    def invoke(self, question: str, context: str = "") -> str:
        """
        Process a question with explicit reasoning steps.
        
        Args:
            question: The question to answer.
            context: Optional context information.
            
        Returns:
            The reasoned response.
        """
        if not question or not question.strip():
            return "Please provide a valid question."
        
        try:
            response = self.chain.invoke({
                "question": question,
                "context": context or "No additional context provided."
            })
            return response
        except Exception as e:
            logger.error(f"SimpleReActAgent error: {e}")
            raise
