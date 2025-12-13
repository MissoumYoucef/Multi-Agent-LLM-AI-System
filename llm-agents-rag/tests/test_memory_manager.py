"""
Unit tests for MemoryManager module.

Tests unified memory management functionality.
"""
import pytest
import os
from unittest.mock import patch, MagicMock

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestMemoryManagerInit:
    """Tests for MemoryManager initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager()
        assert manager.buffer is not None
    
    def test_initialization_with_summarization(self):
        """Test initialization with summarization enabled."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=True)
        assert manager.summary_memory is not None
    
    def test_initialization_without_summarization(self):
        """Test initialization without summarization."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False)
        assert manager.summary_memory is None
    
    def test_initialization_without_persistence(self):
        """Test initialization without persistence."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_persistence=False)
        assert manager.store is None


class TestAddMessages:
    """Tests for adding messages."""
    
    def test_add_user_message(self):
        """Test adding user message."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False)
        msg = manager.add_user_message("Hello!")
        
        assert msg.role == "user"
        assert msg.content == "Hello!"
    
    def test_add_assistant_message(self):
        """Test adding assistant message."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False)
        msg = manager.add_assistant_message("Hi there!")
        
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"
    
    def test_add_message_with_metadata(self):
        """Test adding message with metadata."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False)
        msg = manager.add_user_message(
            "Hello!",
            metadata={"source": "api"}
        )
        
        assert msg.metadata["source"] == "api"


class TestGetContext:
    """Tests for get_context method."""
    
    def test_get_context(self):
        """Test getting context."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False)
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there")
        
        context = manager.get_context()
        assert "Hello" in context
        assert "Hi there" in context
    
    def test_get_context_with_limit(self):
        """Test getting context with message limit."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False)
        manager.add_user_message("Msg 1")
        manager.add_assistant_message("Msg 2")
        manager.add_user_message("Msg 3")
        
        context = manager.get_context(max_messages=2)
        assert "Msg 3" in context


class TestGetMessages:
    """Tests for get_messages method."""
    
    def test_get_messages(self):
        """Test getting message objects."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False)
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        
        messages = manager.get_messages()
        assert len(messages) == 2


class TestSessionManagement:
    """Tests for session management."""
    
    def test_clear_session(self):
        """Test clearing session."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False, enable_persistence=False)
        manager.add_user_message("Hello")
        manager.clear_session()
        
        messages = manager.get_messages()
        assert len(messages) == 0
    
    def test_list_sessions(self):
        """Test listing sessions."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False)
        manager.add_user_message("Hello", session_id="session_a")
        manager.add_user_message("Hi", session_id="session_b")
        
        sessions = manager.list_sessions()
        assert "session_a" in sessions
        assert "session_b" in sessions
    
    def test_get_session_stats(self):
        """Test getting session stats."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False)
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        
        stats = manager.get_session_stats()
        assert stats["message_count"] == 2


class TestPersistence:
    """Tests for session persistence."""
    
    def test_save_session(self):
        """Test saving session."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False, enable_persistence=True)
        manager.add_user_message("Hello")
        
        result = manager.save_session()
        assert result is True
    
    def test_save_session_no_persistence(self):
        """Test save when persistence disabled."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False, enable_persistence=False)
        manager.add_user_message("Hello")
        
        result = manager.save_session()
        assert result is False
    
    def test_load_session_no_persistence(self):
        """Test load when persistence disabled."""
        from src.memory.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_summarization=False, enable_persistence=False)
        result = manager.load_session()
        assert result is False


class TestMemoryContext:
    """Tests for MemoryContext context manager."""
    
    def test_context_manager(self):
        """Test using MemoryContext."""
        from src.memory.memory_manager import MemoryManager, MemoryContext
        
        manager = MemoryManager(enable_summarization=False)
        
        with MemoryContext(manager, "test_session") as mem:
            mem.add_user_message("Hello", session_id="test_session")
            
        # Session should be saved on exit


class TestCreateMemoryManager:
    """Tests for create_memory_manager factory function."""
    
    def test_factory_function(self):
        """Test factory function."""
        from src.memory.memory_manager import create_memory_manager
        
        manager = create_memory_manager(buffer_size=5)
        assert manager.buffer.max_messages == 5
    
    def test_factory_with_summarization(self):
        """Test factory with summarization."""
        from src.memory.memory_manager import create_memory_manager
        
        manager = create_memory_manager(enable_summarization=True)
        assert manager.summary_memory is not None
