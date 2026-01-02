"""
Analyzer Agent module.

Provides an analytical agent that reviews solutions for accuracy and completeness.
"""
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.config import GOOGLE_API_KEY, LLM_MODEL, USE_LOCAL, LOCAL_LLM_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class AnalyzerAgent:
    """
    An analytical agent that reviews solutions for accuracy and completeness.

    Verifies solutions against the original context.
    """

    def __init__(self, model: str = None):
        """
        Initialize the analyzer agent.

        Args:
            model: Optional model name override. Defaults to LLM_MODEL from config.
        """
        self.model_name = model or LLM_MODEL
        if USE_LOCAL:
            logger.info(f"Using local LLM for analyzer: {LOCAL_LLM_MODEL}")
            self.llm = ChatOllama(
                model=LOCAL_LLM_MODEL,
                base_url=OLLAMA_BASE_URL
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=GOOGLE_API_KEY
            )
        self.prompt = PromptTemplate(
            template="""You are an analytical assistant.
            Analyze the provided solution and context to ensure accuracy and completeness.

            Problem: {problem}
            Solution: {solution}
            Context: {context}

            Analysis:""",
            input_variables=["problem", "solution", "context"]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info(f"AnalyzerAgent initialized with model: {self.model_name}")

    def invoke(self, problem: str, solution: str, context: str) -> str:
        """
        Analyze a solution for accuracy and completeness.

        Args:
            problem: The original problem.
            solution: The proposed solution.
            context: The context used to generate the solution.

        Returns:
            An analysis of the solution.

        Raises:
            Exception: If the LLM call fails.
        """
        if not solution or not solution.strip():
            logger.warning("Empty solution received for analysis")
            return "No solution provided to analyze."

        try:
            response = self.chain.invoke({
                "problem": problem or "",
                "solution": solution,
                "context": context or ""
            })
            logger.debug(f"Generated analysis for problem: {problem[:50] if problem else 'N/A'}...")
            return response
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            raise

