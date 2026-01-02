"""
Memory Manager module.

Provides a unified interface for managing conversation memory,
combining short-term buffer with long-term summarization.
"""
import logging
from typing import Dict, List, Optional, Any

from .conversation_memory import (
    ConversationBuffer,
    ConversationSummaryMemory,
    MessageStore,
    Message
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified memory manager for conversation handling.

    Combines:
    - Short-term memory (recent messages buffer)
    - Long-term memory (summarized history)
    - Persistent storage (session save/load)
    """

    def __init__(
        self,
        buffer_size: int = 10,
        summary_threshold: int = 20,
        session_ttl: int = 3600,
        enable_summarization: bool = True,
        enable_persistence: bool = True
    ):
        """
        Initialize memory manager.

        Args:
            buffer_size: Max messages in active buffer.
            summary_threshold: Trigger summarization at this count.
            session_ttl: Session time-to-live in seconds.
            enable_summarization: Whether to enable LLM summarization.
            enable_persistence: Whether to enable session persistence.
        """
        self.buffer = ConversationBuffer(
            max_messages=buffer_size,
            summary_threshold=summary_threshold,
            ttl_seconds=session_ttl
        )

        self.enable_summarization = enable_summarization
        self.enable_persistence = enable_persistence

        if enable_summarization:
            self.summary_memory = ConversationSummaryMemory()
        else:
            self.summary_memory = None

        if enable_persistence:
            self.store = MessageStore(default_ttl=session_ttl * 24)  # 24x TTL for storage
        else:
            self.store = None

        logger.info(f"MemoryManager initialized: buffer={buffer_size}, "
                   f"summarization={enable_summarization}, persistence={enable_persistence}")

    def add_user_message(
        self,
        content: str,
        session_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add a user message to memory."""
        message = self.buffer.add_message(
            role="user",
            content=content,
            session_id=session_id,
            metadata=metadata
        )
        self._check_summarization(session_id)
        return message

    def add_assistant_message(
        self,
        content: str,
        session_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add an assistant message to memory."""
        message = self.buffer.add_message(
            role="assistant",
            content=content,
            session_id=session_id,
            metadata=metadata
        )
        self._check_summarization(session_id)
        return message

    def get_context(
        self,
        session_id: str = "default",
        max_messages: Optional[int] = None,
        include_summary: bool = True
    ) -> str:
        """
        Get formatted conversation context for LLM prompts.

        Args:
            session_id: Session identifier.
            max_messages: Maximum recent messages to include.
            include_summary: Whether to include historical summary.

        Returns:
            Formatted context string.
        """
        return self.buffer.get_formatted_history(
            session_id=session_id,
            limit=max_messages
        )

    def get_messages(
        self,
        session_id: str = "default",
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get message objects from memory."""
        return self.buffer.get_history(
            session_id=session_id,
            limit=limit,
            include_summary=False
        )

    def get_summary(self, session_id: str = "default") -> str:
        """Get the conversation summary."""
        return self.buffer.get_summary(session_id)

    def clear_session(self, session_id: str = "default") -> None:
        """Clear a session's memory."""
        self.buffer.clear_session(session_id)
        if self.store:
            self.store.delete_session(session_id)
        logger.info(f"Cleared memory for session: {session_id}")

    def save_session(self, session_id: str = "default") -> bool:
        """
        Save session to persistent storage.

        Returns:
            True if saved successfully, False otherwise.
        """
        if not self.store:
            logger.warning("Persistence not enabled")
            return False

        try:
            messages = self.buffer.get_history(session_id, include_summary=False)
            summary = self.buffer.get_summary(session_id)
            self.store.save_session(session_id, messages, summary)
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False

    def load_session(self, session_id: str = "default") -> bool:
        """
        Load session from persistent storage.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.store:
            logger.warning("Persistence not enabled")
            return False

        try:
            data = self.store.load_session(session_id)
            if not data:
                return False

            # Restore messages to buffer
            for msg in data["messages"]:
                self.buffer.add_message(
                    role=msg.role,
                    content=msg.content,
                    session_id=session_id,
                    metadata=msg.metadata
                )

            # Restore summary
            if data.get("summary"):
                self.buffer.set_summary(data["summary"], session_id)

            logger.info(f"Loaded session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False

    def get_session_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """Get statistics about a session."""
        return {
            "message_count": self.buffer.get_message_count(session_id),
            "has_summary": bool(self.buffer.get_summary(session_id)),
            "needs_summarization": self.buffer.needs_summarization(session_id)
        }

    def list_sessions(self) -> List[str]:
        """List all active sessions."""
        return self.buffer.get_all_session_ids()

    def _check_summarization(self, session_id: str) -> None:
        """Check and trigger summarization if needed."""
        if not self.enable_summarization or not self.summary_memory:
            return

        if self.summary_memory.should_summarize(self.buffer, session_id):
            try:
                messages = self.buffer.get_history(session_id, include_summary=False)
                # Summarize older messages (all but last few)
                to_summarize = messages[:-5] if len(messages) > 5 else messages

                if to_summarize:
                    new_summary = self.summary_memory.summarize(to_summarize)
                    existing = self.buffer.get_summary(session_id)

                    if existing:
                        combined = f"{existing}\n\nLater: {new_summary}"
                    else:
                        combined = new_summary

                    self.buffer.set_summary(combined, session_id)
                    logger.info(f"Updated summary for session {session_id}")

            except Exception as e:
                logger.error(f"Summarization failed for {session_id}: {e}")


class MemoryContext:
    """
    Context manager for memory operations.

    Automatically saves session on exit.
    """

    def __init__(self, manager: MemoryManager, session_id: str = "default"):
        """Initialize memory context."""
        self.manager = manager
        self.session_id = session_id

    def __enter__(self) -> MemoryManager:
        """Enter context and optionally load session."""
        self.manager.load_session(self.session_id)
        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and save session."""
        self.manager.save_session(self.session_id)


# Factory function for easy creation
def create_memory_manager(
    buffer_size: int = 10,
    enable_summarization: bool = False,  # Default off for cost savings
    enable_persistence: bool = True
) -> MemoryManager:
    """
    Create a memory manager with sensible defaults.

    Args:
        buffer_size: Messages to keep in buffer.
        enable_summarization: Enable LLM-based summarization.
        enable_persistence: Enable session persistence.

    Returns:
        Configured MemoryManager instance.
    """
    return MemoryManager(
        buffer_size=buffer_size,
        enable_summarization=enable_summarization,
        enable_persistence=enable_persistence
    )
