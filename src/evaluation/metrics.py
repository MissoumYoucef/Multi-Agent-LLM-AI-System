"""
Evaluation Metrics module.

Provides metrics for evaluating RAG system outputs.
"""
import logging
from difflib import SequenceMatcher
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain.prompts import PromptTemplate ---> error

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.config import GOOGLE_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    A collection of metrics for evaluating LLM responses.
    
    Includes:
    - Functional correctness (keyword matching)
    - Lexical exactness (sequence matching)
    - ROUGE-L score (longest common subsequence)
    - AI Judge (LLM-based evaluation)
    """
    
    def __init__(self, model: str = None):
        """
        Initialize the evaluation metrics.
        
        Args:
            model: Optional model name override. Defaults to LLM_MODEL from config.
        """
        self.model_name = model or LLM_MODEL
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=GOOGLE_API_KEY
        )
        logger.info(f"EvaluationMetrics initialized with model: {self.model_name}")

    def functional_correctness(self, prediction: str, expected_keywords: List[str]) -> float:
        """
        Check if expected keywords are present in the prediction.
        
        Args:
            prediction: The model's prediction.
            expected_keywords: List of keywords expected in the prediction.
            
        Returns:
            A score from 0.0 to 1.0 indicating keyword coverage.
        """
        if not expected_keywords:
            return 1.0
        
        if not prediction:
            return 0.0
        
        prediction_lower = prediction.lower()
        matches = sum(1 for keyword in expected_keywords 
                     if keyword.lower() in prediction_lower)
        score = matches / len(expected_keywords)
        logger.debug(f"Functional correctness: {matches}/{len(expected_keywords)} = {score:.2f}")
        return score

    def lexical_exactness(self, prediction: str, reference: str) -> float:
        """
        Compute similarity ratio between prediction and reference.
        
        Uses SequenceMatcher for fuzzy string matching.
        
        Args:
            prediction: The model's prediction.
            reference: The reference/ground truth answer.
            
        Returns:
            A similarity score from 0.0 to 1.0.
        """
        if not prediction or not reference:
            return 0.0
        
        score = SequenceMatcher(None, prediction, reference).ratio()
        logger.debug(f"Lexical exactness: {score:.2f}")
        return score

    def rouge_l_score(self, prediction: str, reference: str) -> float:
        """
        Compute ROUGE-L score based on Longest Common Subsequence.
        
        Args:
            prediction: The model's prediction.
            reference: The reference/ground truth answer.
            
        Returns:
            F1-based ROUGE-L score from 0.0 to 1.0.
        """
        if not prediction or not reference:
            return 0.0
        
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        # Compute LCS length
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        logger.debug(f"ROUGE-L score: {f1:.2f}")
        return f1

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute the length of the Longest Common Subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]

    def ai_judge(
        self, 
        question: str, 
        prediction: str, 
        reference: str,
        max_retries: int = 3
    ) -> float:
        """
        Use an LLM to score the prediction against the reference.
        
        Args:
            question: The original question.
            prediction: The model's prediction.
            reference: The reference/ground truth answer.
            max_retries: Maximum number of retry attempts on failure.
            
        Returns:
            A normalized score from 0.0 to 1.0.
        """
        prompt = PromptTemplate(
            template="""You are an impartial judge. Evaluate the quality of the AI's answer.
            
            Question: {question}
            Ground Truth: {reference}
            AI Answer: {prediction}
            
            Score the AI Answer from 0 to 10 based on accuracy and completeness relative to the Ground Truth.
            Return ONLY the number, nothing else.
            """,
            input_variables=["question", "reference", "prediction"]
        )
        chain = prompt | self.llm | StrOutputParser()
        
        for attempt in range(max_retries):
            try:
                score_str = chain.invoke({
                    "question": question,
                    "reference": reference,
                    "prediction": prediction
                })
                # Parse the score, handling potential whitespace and text
                score_str = score_str.strip()
                # Extract just the number if there's extra text
                for word in score_str.split():
                    try:
                        score = float(word)
                        if 0 <= score <= 10:
                            normalized = score / 10.0
                            logger.debug(f"AI Judge score: {score}/10 = {normalized:.2f}")
                            return normalized
                    except ValueError:
                        continue
                
                logger.warning(f"Could not parse AI Judge response: {score_str}")
            except Exception as e:
                logger.warning(f"AI Judge attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"AI Judge failed after {max_retries} attempts")
        
        return 0.0

