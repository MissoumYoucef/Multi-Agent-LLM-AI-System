from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.config import GOOGLE_API_KEY

class AnalyzerAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
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

    def invoke(self, problem: str, solution: str, context: str):
        return self.chain.invoke({"problem": problem, "solution": solution, "context": context})
