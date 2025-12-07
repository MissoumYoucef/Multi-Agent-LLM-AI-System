from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.config import GOOGLE_API_KEY

class SolverAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
        self.prompt = PromptTemplate(
            template="""You are an expert problem solver. 
            Solve the following problem step-by-step.
            
            Problem: {problem}
            Context: {context}
            
            Solution:""",
            input_variables=["problem", "context"]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, problem: str, context: str):
        return self.chain.invoke({"problem": problem, "context": context})
