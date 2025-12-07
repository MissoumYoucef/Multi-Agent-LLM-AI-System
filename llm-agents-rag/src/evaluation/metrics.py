from difflib import SequenceMatcher
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.config import GOOGLE_API_KEY

class EvaluationMetrics:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

    def functional_correctness(self, prediction: str, expected_keywords: list) -> float:
        """Checks if all expected keywords are present in the prediction."""
        if not expected_keywords:
            return 1.0
        
        prediction = prediction.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in prediction)
        return matches / len(expected_keywords)

    def lexical_exactness(self, prediction: str, reference: str) -> float:
        """Computes similarity ratio between prediction and reference."""
        return SequenceMatcher(None, prediction, reference).ratio()

    def ai_judge(self, question: str, prediction: str, reference: str) -> float:
        """Uses LLM to score the prediction against the reference."""
        prompt = PromptTemplate(
            template="""You are an impartial judge. Evaluate the quality of the AI's answer.
            
            Question: {question}
            Ground Truth: {reference}
            AI Answer: {prediction}
            
            Score the AI Answer from 0 to 10 based on accuracy and completeness relative to the Ground Truth.
            Return ONLY the number.
            """,
            input_variables=["question", "reference", "prediction"]
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            score_str = chain.invoke({"question": question, "reference": reference, "prediction": prediction})
            score = float(score_str.strip())
            return score / 10.0 # Normalize to 0-1
        except Exception as e:
            print(f"Error in AI Judge: {e}")
            return 0.0
