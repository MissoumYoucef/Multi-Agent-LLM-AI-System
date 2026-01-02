"""
Unit tests for ConversationMemory module.

Tests Message, ConversationBuffer, ConversationSummaryMemory, and MessageStore.
"""
import pytest
import os
import time
from unittest.mock import patch, MagicMock

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestMessage:
    """Tests for Message dataclass."""

    def test_creation(self):
        """Test Message creation."""
        from src.memory.conversation_memory import Message

        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self):
        """Test Message to_dict conversion."""
        from src.memory.conversation_memory import Message

        msg = Message(role="user", content="Hello")
        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert "timestamp" in data

    def test_from_dict(self):
        """Test Message from_dict creation."""
        from src.memory.conversation_memory import Message

        data = {"role": "assistant", "content": "Hi there"}
        msg = Message.from_dict(data)

        assert msg.role == "assistant"
        assert msg.content == "Hi there"


class TestConversationBuffer:
    """Tests for ConversationBuffer class."""

    def test_initialization(self):
        """Test buffer initialization."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer(max_messages=5)
        assert buffer.max_messages == 5

    def test_add_message(self):
        """Test adding messages."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer()
        msg = buffer.add_message("user", "Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_get_history(self):
        """Test getting message history."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer()
        buffer.add_message("user", "Hello")
        buffer.add_message("assistant", "Hi there")

        history = buffer.get_history(include_summary=False)
        assert len(history) == 2

    def test_get_history_with_limit(self):
        """Test getting limited history."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer()
        buffer.add_message("user", "Msg 1")
        buffer.add_message("assistant", "Msg 2")
        buffer.add_message("user", "Msg 3")

        history = buffer.get_history(limit=2, include_summary=False)
        assert len(history) == 2

    def test_max_messages_eviction(self):
        """Test message eviction when max reached."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer(max_messages=2)
        buffer.add_message("user", "Msg 1")
        buffer.add_message("assistant", "Msg 2")
        buffer.add_message("user", "Msg 3")

        history = buffer.get_history(include_summary=False)
        assert len(history) == 2
        assert history[-1].content == "Msg 3"

    def test_get_formatted_history(self):
        """Test getting formatted history string."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer()
        buffer.add_message("user", "Hello")
        buffer.add_message("assistant", "Hi there")

        formatted = buffer.get_formatted_history()
        assert "User: Hello" in formatted
        assert "Assistant: Hi there" in formatted

    def test_clear_session(self):
        """Test clearing session."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer()
        buffer.add_message("user", "Hello", session_id="test")
        buffer.clear_session("test")

        history = buffer.get_history(session_id="test", include_summary=False)
        assert len(history) == 0

    def test_get_message_count(self):
        """Test getting message count."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer()
        buffer.add_message("user", "Hello")
        buffer.add_message("assistant", "Hi")

        count = buffer.get_message_count()
        assert count == 2

    def test_needs_summarization(self):
        """Test summarization threshold check."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer(summary_threshold=3)
        buffer.add_message("user", "Msg 1")
        buffer.add_message("assistant", "Msg 2")

        assert buffer.needs_summarization() is False

        buffer.add_message("user", "Msg 3")
        assert buffer.needs_summarization() is True

    def test_get_and_set_summary(self):
        """Test getting and setting summary."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer()
        buffer.set_summary("Test summary")

        summary = buffer.get_summary()
        assert summary == "Test summary"

    def test_multiple_sessions(self):
        """Test multiple sessions are independent."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer()
        buffer.add_message("user", "Hello A", session_id="session_a")
        buffer.add_message("user", "Hello B", session_id="session_b")

        history_a = buffer.get_history(session_id="session_a", include_summary=False)
        history_b = buffer.get_history(session_id="session_b", include_summary=False)

        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0].content == "Hello A"
        assert history_b[0].content == "Hello B"

    def test_get_all_session_ids(self):
        """Test getting all session IDs."""
        from src.memory.conversation_memory import ConversationBuffer

        buffer = ConversationBuffer()
        buffer.add_message("user", "Hello", session_id="session_1")
        buffer.add_message("user", "Hello", session_id="session_2")

        sessions = buffer.get_all_session_ids()
        assert "session_1" in sessions
        assert "session_2" in sessions


class TestConversationSummaryMemory:
    """Tests for ConversationSummaryMemory class."""

    def test_initialization(self):
        """Test initialization without LLM."""
        from src.memory.conversation_memory import ConversationSummaryMemory

        memory = ConversationSummaryMemory(max_summary_tokens=300)
        assert memory.max_summary_tokens == 300

    def test_summarize_empty(self):
        """Test summarizing empty messages."""
        from src.memory.conversation_memory import ConversationSummaryMemory

        memory = ConversationSummaryMemory()
        summary = memory.summarize([])
        assert summary == ""

    def test_should_summarize(self):
        """Test should_summarize check."""
        from src.memory.conversation_memory import (
            ConversationSummaryMemory,
            ConversationBuffer
        )

        buffer = ConversationBuffer(summary_threshold=3)
        memory = ConversationSummaryMemory()

        buffer.add_message("user", "Msg 1")
        assert memory.should_summarize(buffer, "default") is False

        buffer.add_message("assistant", "Msg 2")
        buffer.add_message("user", "Msg 3")
        assert memory.should_summarize(buffer, "default") is True


class TestMessageStore:
    """Tests for MessageStore class."""

    def test_initialization(self):
        """Test store initialization."""
        from src.memory.conversation_memory import MessageStore

        store = MessageStore(default_ttl=7200)
        assert store.default_ttl == 7200

    def test_save_and_load_session(self):
        """Test saving and loading session."""
        from src.memory.conversation_memory import MessageStore, Message

        store = MessageStore()
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there")
        ]

        store.save_session("test_session", messages, "test summary")

        data = store.load_session("test_session")
        assert data is not None
        assert len(data["messages"]) == 2
        assert data["summary"] == "test summary"

    def test_load_nonexistent_session(self):
        """Test loading nonexistent session."""
        from src.memory.conversation_memory import MessageStore

        store = MessageStore()
        data = store.load_session("nonexistent")
        assert data is None

    def test_delete_session(self):
        """Test deleting session."""
        from src.memory.conversation_memory import MessageStore, Message

        store = MessageStore()
        messages = [Message(role="user", content="Hello")]
        store.save_session("test_session", messages)

        result = store.delete_session("test_session")
        assert result is True
        assert store.load_session("test_session") is None

    def test_delete_nonexistent_session(self):
        """Test deleting nonexistent session."""
        from src.memory.conversation_memory import MessageStore

        store = MessageStore()
        result = store.delete_session("nonexistent")
        assert result is False

    def test_list_sessions(self):
        """Test listing sessions."""
        from src.memory.conversation_memory import MessageStore, Message

        store = MessageStore()
        store.save_session("session_1", [Message(role="user", content="Hi")])
        store.save_session("session_2", [Message(role="user", content="Hello")])

        sessions = store.list_sessions()
        assert "session_1" in sessions
        assert "session_2" in sessions

    def test_session_expiration(self):
        """Test session expiration."""
        from src.memory.conversation_memory import MessageStore, Message

        store = MessageStore(default_ttl=1)
        messages = [Message(role="user", content="Hello")]
        store.save_session("test_session", messages)

        # Manually expire the session
        store._store["test_session"]["expires_at"] = time.time() - 10

        data = store.load_session("test_session")
        assert data is None
