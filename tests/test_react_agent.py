"""
Tests for ReAct Agent module.
"""
import pytest
import os
from unittest.mock import patch, MagicMock

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestSimpleReActAgent:
    """Tests for SimpleReActAgent class."""
    
    def test_initialization(self):
        """Test agent initializes correctly."""
        from src.agents.react_agent import SimpleReActAgent
        
        with patch('src.agents.react_agent.ChatGoogleGenerativeAI') as mock_google, \
             patch('src.agents.react_agent.ChatOllama') as mock_ollama:
            agent = SimpleReActAgent()
            
            assert agent.prompt is not None
            assert agent.chain is not None
            assert mock_google.called or mock_ollama.called
    
    def test_custom_model(self):
        """Test agent accepts custom model."""
        from src.agents.react_agent import SimpleReActAgent
        
        with patch('src.agents.react_agent.ChatGoogleGenerativeAI'), \
             patch('src.agents.react_agent.ChatOllama'):
            agent = SimpleReActAgent(model="gemini-1.5-pro")
            
            assert agent.model_name == "gemini-1.5-pro"
    
    def test_invoke_empty_question(self):
        """Test invoke handles empty question."""
        from src.agents.react_agent import SimpleReActAgent
        
        with patch('src.agents.react_agent.ChatGoogleGenerativeAI'), \
             patch('src.agents.react_agent.ChatOllama'):
            agent = SimpleReActAgent()
            
            result = agent.invoke("")
            assert "valid question" in result.lower()
    
    def test_invoke_calls_chain(self):
        """Test invoke calls the chain correctly."""
        from src.agents.react_agent import SimpleReActAgent
        
        with patch('src.agents.react_agent.ChatGoogleGenerativeAI'), \
             patch('src.agents.react_agent.ChatOllama'):
            agent = SimpleReActAgent()
            agent.chain = MagicMock()
            agent.chain.invoke.return_value = "Step 1: Think. Step 2: Answer."
            
            result = agent.invoke("What is 2+2?", "Math context")
            
            assert "Step" in result
            agent.chain.invoke.assert_called_once()


class TestReActAgent:
    """Tests for ReActAgent class (full agent with tools)."""
    
    def test_initialization(self):
        """Test full ReActAgent initializes."""
        from src.agents.react_agent import ReActAgent
        
        with patch('src.agents.react_agent.ChatGoogleGenerativeAI'), \
             patch('src.agents.react_agent.ChatOllama'):
            with patch('src.agents.react_agent.create_react_agent') as mock_create:
                with patch('src.agents.react_agent.AgentExecutor') as mock_executor:
                    mock_create.return_value = MagicMock()
                    agent = ReActAgent()
                    
                    assert agent.tools is not None
                    assert len(agent.tools) >= 1
    
    def test_extract_reasoning(self):
        """Test reasoning extraction from steps."""
        from src.agents.react_agent import ReActAgent
        
        with patch('src.agents.react_agent.ChatGoogleGenerativeAI'), \
             patch('src.agents.react_agent.ChatOllama'):
            with patch('src.agents.react_agent.create_react_agent') as mock_create:
                with patch('src.agents.react_agent.AgentExecutor'):
                    mock_create.return_value = MagicMock()
                    agent = ReActAgent()
                
                # Mock action
                mock_action = MagicMock()
                mock_action.tool = "calculate"
                mock_action.tool_input = "2+2"
                mock_action.log = "I should calculate this"
                
                steps = [(mock_action, "4")]
                reasoning = agent._extract_reasoning(steps)
                
                assert len(reasoning) == 1
                assert reasoning[0]["action"] == "calculate"
                assert reasoning[0]["observation"] == "4"
    
    def test_get_reasoning_trace_empty(self):
        """Test reasoning trace with no steps."""
        from src.agents.react_agent import ReActAgent
        
        with patch('src.agents.react_agent.ChatGoogleGenerativeAI'), \
             patch('src.agents.react_agent.ChatOllama'):
            with patch('src.agents.react_agent.create_react_agent') as mock_create:
                with patch('src.agents.react_agent.AgentExecutor'):
                    mock_create.return_value = MagicMock()
                    agent = ReActAgent()
                
                result = {"reasoning": []}
                trace = agent.get_reasoning_trace(result)
                
                assert "no intermediate" in trace.lower()
