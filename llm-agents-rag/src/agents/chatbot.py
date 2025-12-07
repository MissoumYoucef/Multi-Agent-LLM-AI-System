from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.config import GOOGLE_API_KEY

class ChatbotAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
        self.prompt = PromptTemplate(
            template="""You are a helpful assistant. Answer the user's question based on your general knowledge.
            If the question requires specific context from documents, say "I need to look that up".
            
            Question: {question}
            Answer:""",
            input_variables=["question"]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, question: str):
        return self.chain.invoke({"question": question})
