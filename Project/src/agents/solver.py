"""
Solver Agent module.

Provides a problem-solving agent that generates step-by-step solutions.
"""
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from ..utils.config import GOOGLE_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


class SolverAgent:
    """
    A problem-solving agent that generates step-by-step solutions.
    
    Uses retrieved context to provide accurate answers.
    """
    
    def __init__(self, model: str = None):
        """
        Initialize the solver agent.
        
        Args:
            model: Optional model name override. Defaults to LLM_MODEL from config.
        """
        self.model_name = model or LLM_MODEL
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=GOOGLE_API_KEY
        )
        self.prompt = PromptTemplate(
            template="""You are an expert problem solver. 
            Solve the following problem step-by-step.
            
            Problem: {problem}
            Context: {context}
            
            Solution:""",
            input_variables=["problem", "context"]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info(f"SolverAgent initialized with model: {self.model_name}")

    def invoke(self, problem: str, context: str) -> str:
        """
        Generate a solution for the given problem.
        
        Args:
            problem: The problem to solve.
            context: Retrieved context to inform the solution.
            
        Returns:
            A step-by-step solution.
            
        Raises:
            Exception: If the LLM call fails.
        """
        if not problem or not problem.strip():
            logger.warning("Empty problem received")
            return "Please provide a valid problem."
        
        try:
            response = self.chain.invoke({"problem": problem, "context": context or ""})
            logger.debug(f"Generated solution for problem: {problem[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            raise

