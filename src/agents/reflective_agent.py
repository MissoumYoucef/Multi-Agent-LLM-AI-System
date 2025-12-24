"""
Reflective Agent module.

Implements self-reflection capability for quality improvement.
The agent evaluates its own output and can refine responses.
"""
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.config import GOOGLE_API_KEY, LLM_MODEL, USE_LOCAL, LOCAL_LLM_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Result of a self-reflection evaluation."""
    quality_score: float  # 0.0 to 1.0
    is_acceptable: bool
    critique: str
    suggestions: str
    refined_response: Optional[str] = None


class ReflectiveAgent:
    """
    Self-reflecting agent that evaluates and improves its responses.
    
    Implements a critique -> refine loop for quality improvement.
    """
    
    def __init__(
        self,
        model: str = None,
        quality_threshold: float = 0.7,
        max_refinements: int = 2
    ):
        """
        Initialize the reflective agent.
        
        Args:
            model: Optional model name override.
            quality_threshold: Minimum quality score to accept (0.0-1.0).
            max_refinements: Maximum number of refinement iterations.
        """
        self.model_name = model or LLM_MODEL
        self.quality_threshold = quality_threshold
        self.max_refinements = max_refinements
        
        if USE_LOCAL:
            logger.info(f"Using local LLM for ReflectiveAgent: {LOCAL_LLM_MODEL}")
            self.llm = ChatOllama(
                model=LOCAL_LLM_MODEL,
                base_url=OLLAMA_BASE_URL
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=GOOGLE_API_KEY
            )
        
        # Critique prompt
        self.critique_prompt = PromptTemplate(
            template="""Evaluate the following response to a question. 
Provide a quality score from 0.0 to 1.0 and constructive feedback.

Question: {question}
Context: {context}
Response: {response}

Evaluate on these criteria:
1. Accuracy: Is the information correct given the context?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it well-structured and easy to understand?
4. Relevance: Does it stay on topic?

Format your evaluation as:
SCORE: [0.0-1.0]
CRITIQUE: [Your critique]
SUGGESTIONS: [Specific improvements]""",
            input_variables=["question", "context", "response"]
        )
        
        # Refinement prompt
        self.refine_prompt = PromptTemplate(
            template="""Improve the following response based on the feedback provided.

                Original Question: {question}
                Context: {context}
                Original Response: {response}
                Critique: {critique}
                Suggestions: {suggestions}

                Write an improved response that addresses the feedback:""",
                input_variables=["question", "context", "response", "critique", "suggestions"]
            )
        
        self.critique_chain = self.critique_prompt | self.llm | StrOutputParser()
        self.refine_chain = self.refine_prompt | self.llm | StrOutputParser()
        
        logger.info(f"ReflectiveAgent initialized with threshold: {quality_threshold}")

    def evaluate(
        self,
        question: str,
        response: str,
        context: str = ""
    ) -> ReflectionResult:
        """
        Evaluate a response quality.
        
        Args:
            question: The original question.
            response: The response to evaluate.
            context: Optional context used for the response.
            
        Returns:
            ReflectionResult with quality assessment.
        """
        if not response or not response.strip():
            return ReflectionResult(
                quality_score=0.0,
                is_acceptable=False,
                critique="Empty response provided.",
                suggestions="Provide a substantive response to the question."
            )
        
        try:
            # Get critique
            critique_output = self.critique_chain.invoke({
                "question": question,
                "context": context or "No additional context.",
                "response": response
            })
            
            # Parse the critique
            score, critique, suggestions = self._parse_critique(critique_output)
            
            return ReflectionResult(
                quality_score=score,
                is_acceptable=score >= self.quality_threshold,
                critique=critique,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            # Return neutral result on error
            return ReflectionResult(
                quality_score=0.5,
                is_acceptable=True,  # Don't block on evaluation errors
                critique=f"Evaluation failed: {str(e)}",
                suggestions="Unable to provide suggestions."
            )

    def _parse_critique(self, critique_text: str) -> Tuple[float, str, str]:
        """
        Parse the critique output to extract score, critique, and suggestions.
        
        Args:
            critique_text: Raw critique output.
            
        Returns:
            Tuple of (score, critique, suggestions).
        """
        lines = critique_text.strip().split('\n')
        
        score = 0.5  # Default
        critique = ""
        suggestions = ""
        
        current_section = None
        
        for line in lines:
            line_upper = line.upper().strip()
            
            if line_upper.startswith('SCORE:'):
                try:
                    score_str = line.split(':', 1)[1].strip()
                    # Extract number from string
                    import re
                    match = re.search(r'(\d+\.?\d*)', score_str)
                    if match:
                        score = float(match.group(1))
                        if score > 1.0:
                            score = score / 10.0  # Handle scores like 8/10
                        score = max(0.0, min(1.0, score))
                except:
                    pass
            elif line_upper.startswith('CRITIQUE:'):
                current_section = 'critique'
                critique = line.split(':', 1)[1].strip() if ':' in line else ""
            elif line_upper.startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
                suggestions = line.split(':', 1)[1].strip() if ':' in line else ""
            elif current_section == 'critique':
                critique += " " + line.strip()
            elif current_section == 'suggestions':
                suggestions += " " + line.strip()
        
        return score, critique.strip(), suggestions.strip()

    def refine(
        self,
        question: str,
        response: str,
        context: str,
        critique: str,
        suggestions: str
    ) -> str:
        """
        Refine a response based on critique and suggestions.
        
        Args:
            question: The original question.
            response: The original response.
            context: The context.
            critique: The critique of the response.
            suggestions: Suggestions for improvement.
            
        Returns:
            The refined response.
        """
        try:
            refined = self.refine_chain.invoke({
                "question": question,
                "context": context or "",
                "response": response,
                "critique": critique,
                "suggestions": suggestions
            })
            return refined
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            return response  # Return original on error

    def reflect_and_improve(
        self,
        question: str,
        response: str,
        context: str = ""
    ) -> Dict:
        """
        Complete reflection loop: evaluate and refine if needed.
        
        Args:
            question: The original question.
            response: The response to potentially improve.
            context: Optional context.
            
        Returns:
            Dictionary with final response, reflection history, and improvements made.
        """
        history = []
        current_response = response
        improvements_made = 0
        
        for iteration in range(self.max_refinements + 1):
            # Evaluate current response
            result = self.evaluate(question, current_response, context)
            
            history.append({
                "iteration": iteration,
                "response_preview": current_response[:200] + "..." if len(current_response) > 200 else current_response,
                "score": result.quality_score,
                "acceptable": result.is_acceptable,
                "critique": result.critique
            })
            
            logger.info(f"Reflection iteration {iteration}: score={result.quality_score:.2f}")
            
            # If acceptable or max iterations reached, stop
            if result.is_acceptable or iteration >= self.max_refinements:
                break
            
            # Refine the response
            current_response = self.refine(
                question, current_response, context,
                result.critique, result.suggestions
            )
            improvements_made += 1
        
        return {
            "final_response": current_response,
            "original_response": response,
            "reflection_history": history,
            "improvements_made": improvements_made,
            "final_score": history[-1]["score"] if history else 0.5
        }


class CriticAgent:
    """
    Standalone critic agent for evaluating other agents' outputs.
    
    Can be used independently of the ReflectiveAgent.
    """
    
    def __init__(self, model: str = None):
        """Initialize the critic agent."""
        self.model_name = model or LLM_MODEL
        if USE_LOCAL:
            logger.info(f"Using local LLM for CriticAgent: {LOCAL_LLM_MODEL}")
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
            template="""As a critic, evaluate this response objectively.

            Question: {question}
            Response: {response}

            Provide:
            1. Strengths (what's good)
            2. Weaknesses (what needs improvement)
            3. Overall assessment (1-2 sentences)

            Your critique:""",
                        input_variables=["question", "response"]
                    )
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def critique(self, question: str, response: str) -> str:
        """
        Provide a critique of a response.
        
        Args:
            question: The original question.
            response: The response to critique.
            
        Returns:
            A structured critique.
        """
        try:
            return self.chain.invoke({"question": question, "response": response})
        except Exception as e:
            logger.error(f"Critique error: {e}")
            return f"Unable to provide critique: {str(e)}"
