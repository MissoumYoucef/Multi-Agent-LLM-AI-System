"""
Conversation Memory module.

Implements conversation storage with sliding window buffer, 
summarization, and session management for multi-turn conversations.
"""
import logging
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in conversation history."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )


class ConversationBuffer:
    """
    In-memory conversation buffer with sliding window.
    
    Maintains the most recent N messages per session,
    with automatic summarization of older messages.
    """
    
    def __init__(
        self,
        max_messages: int = 10,
        summary_threshold: int = 20,
        ttl_seconds: int = 3600
    ):
        """
        Initialize conversation buffer.
        
        Args:
            max_messages: Maximum messages to keep in active buffer.
            summary_threshold: Trigger summarization when history exceeds this.
            ttl_seconds: Time-to-live for sessions in seconds.
        """
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.ttl_seconds = ttl_seconds
        
        # Session storage: {session_id: {"messages": [], "summary": str, "last_access": float}}
        self._sessions: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"messages": [], "summary": "", "last_access": time.time()}
        )
        
        logger.info(f"ConversationBuffer initialized: max_messages={max_messages}, "
                   f"summary_threshold={summary_threshold}")
    
    def add_message(
        self,
        role: str,
        content: str,
        session_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to the conversation buffer.
        
        Args:
            role: Message role ('user' or 'assistant').
            content: Message content.
            session_id: Session identifier.
            metadata: Optional message metadata.
            
        Returns:
            The created Message object.
        """
        self._cleanup_expired_sessions()
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        session = self._sessions[session_id]
        session["messages"].append(message)
        session["last_access"] = time.time()
        
        # Trim to max_messages, keeping most recent
        if len(session["messages"]) > self.max_messages:
            excess = session["messages"][:-self.max_messages]
            session["messages"] = session["messages"][-self.max_messages:]
            
            # Append excess to summary context (simplified)
            if excess:
                excess_text = self._format_messages_for_summary(excess)
                if session["summary"]:
                    session["summary"] += f"\n\n[Earlier conversation]\n{excess_text}"
                else:
                    session["summary"] = f"[Earlier conversation]\n{excess_text}"
        
        logger.debug(f"Added {role} message to session {session_id}")
        return message
    
    def get_history(
        self,
        session_id: str = "default",
        limit: Optional[int] = None,
        include_summary: bool = True
    ) -> List[Message]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier.
            limit: Maximum messages to return (None for all).
            include_summary: Whether to include summary as first message.
            
        Returns:
            List of Message objects.
        """
        if session_id not in self._sessions:
            return []
        
        session = self._sessions[session_id]
        session["last_access"] = time.time()
        
        messages = session["messages"]
        if limit:
            messages = messages[-limit:]
        
        # Prepend summary as system context if available
        result = []
        if include_summary and session["summary"]:
            result.append(Message(
                role="system",
                content=f"Previous conversation summary: {session['summary'][:500]}...",
                metadata={"is_summary": True}
            ))
        
        result.extend(messages)
        return result
    
    def get_formatted_history(
        self,
        session_id: str = "default",
        limit: Optional[int] = None
    ) -> str:
        """
        Get formatted conversation history as a string.
        
        Args:
            session_id: Session identifier.
            limit: Maximum messages to include.
            
        Returns:
            Formatted conversation string.
        """
        messages = self.get_history(session_id, limit, include_summary=True)
        
        if not messages:
            return ""
        
        formatted = []
        for msg in messages:
            role_label = msg.role.capitalize()
            formatted.append(f"{role_label}: {msg.content}")
        
        return "\n".join(formatted)
    
    def get_summary(self, session_id: str = "default") -> str:
        """Get the summary for a session."""
        if session_id not in self._sessions:
            return ""
        return self._sessions[session_id].get("summary", "")
    
    def set_summary(self, summary: str, session_id: str = "default") -> None:
        """Set the summary for a session."""
        self._sessions[session_id]["summary"] = summary
        self._sessions[session_id]["last_access"] = time.time()
    
    def clear_session(self, session_id: str = "default") -> None:
        """Clear a session's history."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
    
    def get_message_count(self, session_id: str = "default") -> int:
        """Get the number of messages in a session."""
        if session_id not in self._sessions:
            return 0
        return len(self._sessions[session_id]["messages"])
    
    def needs_summarization(self, session_id: str = "default") -> bool:
        """Check if session needs summarization based on threshold."""
        return self.get_message_count(session_id) >= self.summary_threshold
    
    def get_all_session_ids(self) -> List[str]:
        """Get all active session IDs."""
        return list(self._sessions.keys())
    
    def _format_messages_for_summary(self, messages: List[Message]) -> str:
        """Format messages for summarization."""
        lines = []
        for msg in messages:
            lines.append(f"{msg.role}: {msg.content[:100]}...")
        return "\n".join(lines)
    
    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions based on TTL."""
        current_time = time.time()
        expired = [
            sid for sid, session in self._sessions.items()
            if current_time - session["last_access"] > self.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.debug(f"Expired session removed: {sid}")


class ConversationSummaryMemory:
    """
    Summarizes conversation history to reduce token usage.
    
    Uses an LLM to generate concise summaries of past conversations.
    """
    
    def __init__(self, llm=None, max_summary_tokens: int = 500):
        """
        Initialize summary memory.
        
        Args:
            llm: Language model for summarization (optional, lazy init).
            max_summary_tokens: Maximum tokens for summary.
        """
        self._llm = llm
        self.max_summary_tokens = max_summary_tokens
    
    @property
    def llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from ..utils.config import GOOGLE_API_KEY, LLM_MODEL
            self._llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                google_api_key=GOOGLE_API_KEY
            )
        return self._llm
    
    def summarize(self, messages: List[Message]) -> str:
        """
        Summarize a list of messages.
        
        Args:
            messages: List of messages to summarize.
            
        Returns:
            Summarized conversation string.
        """
        if not messages:
            return ""
        
        # Format messages for summarization
        conversation = "\n".join([
            f"{msg.role.capitalize()}: {msg.content}"
            for msg in messages
        ])
        
        try:
            prompt = f"""Summarize the following conversation in a concise paragraph.
Focus on key topics discussed, questions asked, and important information exchanged.
Keep the summary under {self.max_summary_tokens} tokens.

Conversation:
{conversation}

Summary:"""
            
            response = self.llm.invoke(prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Generated summary of {len(messages)} messages")
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: simple truncation
            return f"Previous conversation with {len(messages)} messages."
    
    def should_summarize(self, buffer: ConversationBuffer, session_id: str) -> bool:
        """Check if buffer should be summarized."""
        return buffer.needs_summarization(session_id)


class MessageStore:
    """
    Persistent message storage with TTL and session management.
    
    Note: This implementation uses in-memory storage.
    For production, extend to use Redis or a database.
    """
    
    def __init__(self, default_ttl: int = 86400):
        """
        Initialize message store.
        
        Args:
            default_ttl: Default time-to-live in seconds (24 hours).
        """
        self.default_ttl = default_ttl
        self._store: Dict[str, Dict[str, Any]] = {}
    
    def save_session(
        self,
        session_id: str,
        messages: List[Message],
        summary: str = "",
        ttl: Optional[int] = None
    ) -> None:
        """
        Save a session to the store.
        
        Args:
            session_id: Session identifier.
            messages: List of messages.
            summary: Conversation summary.
            ttl: Optional TTL override.
        """
        self._store[session_id] = {
            "messages": [msg.to_dict() for msg in messages],
            "summary": summary,
            "created_at": time.time(),
            "expires_at": time.time() + (ttl or self.default_ttl)
        }
        logger.debug(f"Saved session {session_id} with {len(messages)} messages")
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a session from the store.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Session data or None if not found/expired.
        """
        if session_id not in self._store:
            return None
        
        session = self._store[session_id]
        
        # Check expiration
        if time.time() > session.get("expires_at", 0):
            del self._store[session_id]
            return None
        
        return {
            "messages": [Message.from_dict(m) for m in session["messages"]],
            "summary": session.get("summary", "")
        }
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session from the store."""
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        self._cleanup_expired()
        return list(self._store.keys())
    
    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        current = time.time()
        expired = [
            sid for sid, data in self._store.items()
            if current > data.get("expires_at", 0)
        ]
        for sid in expired:
            del self._store[sid]
