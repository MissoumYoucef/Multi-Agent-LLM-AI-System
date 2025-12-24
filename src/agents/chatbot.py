"""
Chatbot Agent module.

Provides a general-purpose chatbot for handling user queries
with optional conversation memory support.
"""
import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.config import GOOGLE_API_KEY, LLM_MODEL, USE_LOCAL, LOCAL_LLM_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class ChatbotAgent:
    """
    A general-purpose chatbot agent that responds to user questions.
    
    Uses Google's Gemini model for generating responses.
    Supports optional conversation memory for multi-turn interactions.
    """
    
    def __init__(
        self,
        model: str = None,
        memory_manager=None,
        include_history: bool = True,
        max_history_messages: int = 5
    ):
        """
        Initialize the chatbot agent.
        
        Args:
            model: Optional model name override. Defaults to LLM_MODEL from config.
            memory_manager: Optional MemoryManager for conversation history.
            include_history: Whether to include conversation history in prompts.
            max_history_messages: Maximum history messages to include.
        """
        self.model_name = model or LLM_MODEL
        self.memory_manager = memory_manager
        self.include_history = include_history
        self.max_history_messages = max_history_messages
        
        if USE_LOCAL:
            logger.info(f"Using local LLM: {LOCAL_LLM_MODEL}")
            self.llm = ChatOllama(
                model=LOCAL_LLM_MODEL,
                base_url=OLLAMA_BASE_URL
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=GOOGLE_API_KEY
            )
        
        # Prompt without history (backward compatible)
        self.simple_prompt = PromptTemplate(
            template="""You are a helpful assistant. Answer the user's question based on your general knowledge.
If the question requires specific context from documents, say "I need to look that up".

Question: {question}
Answer:""",
            input_variables=["question"]
        )
        
        # Prompt with conversation history
        self.memory_prompt = PromptTemplate(
            template="""You are a helpful assistant. Answer the user's question based on your general knowledge and the conversation history.
If the question requires specific context from documents, say "I need to look that up".

Conversation History:
{history}

Current Question: {question}
Answer:""",
            input_variables=["history", "question"]
        )
        
        self.simple_chain = self.simple_prompt | self.llm | StrOutputParser()
        self.memory_chain = self.memory_prompt | self.llm | StrOutputParser()
        
        logger.info(f"ChatbotAgent initialized with model: {self.model_name}, "
                   f"memory={'enabled' if memory_manager else 'disabled'}")

    def invoke(
        self,
        question: str,
        session_id: str = "default",
        context: Optional[str] = None
    ) -> str:
        """
        Generate a response to the user's question.
        
        Args:
            question: The user's question.
            session_id: Session ID for memory (if memory enabled).
            context: Optional additional context.
            
        Returns:
            The chatbot's response.
            
        Raises:
            Exception: If the LLM call fails.
        """
        if not question or not question.strip():
            logger.warning("Empty question received")
            return "Please provide a valid question."
        
        try:
            # Add user message to memory if enabled
            if self.memory_manager:
                self.memory_manager.add_user_message(question, session_id)
            
            # Generate response
            if self.memory_manager and self.include_history:
                history = self.memory_manager.get_context(
                    session_id=session_id,
                    max_messages=self.max_history_messages
                )
                response = self.memory_chain.invoke({
                    "history": history or "No previous conversation.",
                    "question": question
                })
            else:
                response = self.simple_chain.invoke({"question": question})
            
            # Add assistant response to memory if enabled
            if self.memory_manager:
                self.memory_manager.add_assistant_message(response, session_id)
            
            logger.debug(f"Generated response for question: {question[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def invoke_with_context(
        self,
        question: str,
        context: str,
        session_id: str = "default"
    ) -> str:
        """
        Generate a response with additional context.
        
        Args:
            question: The user's question.
            context: Additional context (e.g., from RAG).
            session_id: Session ID for memory.
            
        Returns:
            The chatbot's response.
        """
        # Create context-aware prompt
        context_prompt = PromptTemplate(
            template="""You are a helpful assistant. Answer the user's question using the provided context.

Context:
{context}

{history_section}

Question: {question}
Answer:""",
            input_variables=["context", "history_section", "question"]
        )
        context_chain = context_prompt | self.llm | StrOutputParser()
        
        try:
            if self.memory_manager:
                self.memory_manager.add_user_message(question, session_id)
                history = self.memory_manager.get_context(
                    session_id=session_id,
                    max_messages=self.max_history_messages
                )
                history_section = f"Conversation History:\n{history}" if history else ""
            else:
                history_section = ""
            
            response = context_chain.invoke({
                "context": context,
                "history_section": history_section,
                "question": question
            })
            
            if self.memory_manager:
                self.memory_manager.add_assistant_message(response, session_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            raise

    def clear_memory(self, session_id: str = "default") -> None:
        """Clear memory for a session."""
        if self.memory_manager:
            self.memory_manager.clear_session(session_id)
            logger.info(f"Cleared memory for session: {session_id}")
