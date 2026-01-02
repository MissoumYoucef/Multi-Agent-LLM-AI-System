"""
Unit tests for Agent classes.

Tests agent initialization and method signatures.
"""
import pytest
import os
from unittest.mock import patch, MagicMock

# Set env var before importing
os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestChatbotAgent:
    """Tests for ChatbotAgent class."""

    def test_initialization(self):
        """Test agent initializes correctly."""
        from src.agents.chatbot import ChatbotAgent

        with patch('src.agents.chatbot.ChatGoogleGenerativeAI') as mock_google, \
             patch('src.agents.chatbot.ChatOllama') as mock_ollama:
            agent = ChatbotAgent()

            assert agent.simple_prompt is not None
            assert agent.simple_chain is not None
            assert mock_google.called or mock_ollama.called

    def test_custom_model(self):
        """Test agent accepts custom model."""
        from src.agents.chatbot import ChatbotAgent

        with patch('src.agents.chatbot.ChatGoogleGenerativeAI'), \
             patch('src.agents.chatbot.ChatOllama'):
            agent = ChatbotAgent(model="gemini-1.5-pro")

            assert agent.model_name == "gemini-1.5-pro"

    def test_invoke_empty_question(self):
        """Test invoke handles empty question."""
        from src.agents.chatbot import ChatbotAgent

        with patch('src.agents.chatbot.ChatGoogleGenerativeAI'), \
             patch('src.agents.chatbot.ChatOllama'):
            agent = ChatbotAgent()

            result = agent.invoke("")
            assert "valid question" in result.lower()

    def test_invoke_calls_chain(self):
        """Test invoke calls the chain correctly."""
        from src.agents.chatbot import ChatbotAgent

        with patch('src.agents.chatbot.ChatGoogleGenerativeAI'), \
             patch('src.agents.chatbot.ChatOllama'):
            agent = ChatbotAgent()
            agent.simple_chain = MagicMock()
            agent.simple_chain.invoke.return_value = "Test response"

            result = agent.invoke("What is AI?")

            assert result == "Test response"
            agent.simple_chain.invoke.assert_called_once()


class TestSolverAgent:
    """Tests for SolverAgent class."""

    def test_initialization(self):
        """Test agent initializes correctly."""
        from src.agents.solver import SolverAgent

        with patch('src.agents.solver.ChatGoogleGenerativeAI') as mock_google, \
             patch('src.agents.solver.ChatOllama') as mock_ollama:
            agent = SolverAgent()

            assert agent.prompt is not None
            assert agent.chain is not None
            assert mock_google.called or mock_ollama.called

    def test_custom_model(self):
        """Test agent accepts custom model."""
        from src.agents.solver import SolverAgent

        with patch('src.agents.solver.ChatGoogleGenerativeAI'), \
             patch('src.agents.solver.ChatOllama'):
            agent = SolverAgent(model="gemini-1.5-flash")

            assert agent.model_name == "gemini-1.5-flash"

    def test_invoke_empty_problem(self):
        """Test invoke handles empty problem."""
        from src.agents.solver import SolverAgent

        with patch('src.agents.solver.ChatGoogleGenerativeAI'), \
             patch('src.agents.solver.ChatOllama'):
            agent = SolverAgent()

            result = agent.invoke("", "some context")
            assert "valid problem" in result.lower()

    def test_invoke_calls_chain(self):
        """Test invoke calls the chain correctly."""
        from src.agents.solver import SolverAgent

        with patch('src.agents.solver.ChatGoogleGenerativeAI'), \
             patch('src.agents.solver.ChatOllama'):
            agent = SolverAgent()
            agent.chain = MagicMock()
            agent.chain.invoke.return_value = "Solution here"

            result = agent.invoke("Solve X", "Context about X")

            assert result == "Solution here"
            agent.chain.invoke.assert_called_once()


class TestAnalyzerAgent:
    """Tests for AnalyzerAgent class."""

    def test_initialization(self):
        """Test agent initializes correctly."""
        from src.agents.analyzer import AnalyzerAgent

        with patch('src.agents.analyzer.ChatGoogleGenerativeAI') as mock_google, \
             patch('src.agents.analyzer.ChatOllama') as mock_ollama:
            agent = AnalyzerAgent()

            assert agent.prompt is not None
            assert agent.chain is not None
            assert mock_google.called or mock_ollama.called

    def test_custom_model(self):
        """Test agent accepts custom model."""
        from src.agents.analyzer import AnalyzerAgent

        with patch('src.agents.analyzer.ChatGoogleGenerativeAI'), \
             patch('src.agents.analyzer.ChatOllama'):
            agent = AnalyzerAgent(model="gemini-pro-vision")

            assert agent.model_name == "gemini-pro-vision"

    def test_invoke_empty_solution(self):
        """Test invoke handles empty solution."""
        from src.agents.analyzer import AnalyzerAgent

        with patch('src.agents.analyzer.ChatGoogleGenerativeAI'), \
             patch('src.agents.analyzer.ChatOllama'):
            agent = AnalyzerAgent()

            result = agent.invoke("problem", "", "context")
            assert "no solution" in result.lower()

    def test_invoke_calls_chain(self):
        """Test invoke calls the chain correctly."""
        from src.agents.analyzer import AnalyzerAgent

        with patch('src.agents.analyzer.ChatGoogleGenerativeAI'), \
             patch('src.agents.analyzer.ChatOllama'):
            agent = AnalyzerAgent()
            agent.chain = MagicMock()
            agent.chain.invoke.return_value = "Analysis complete"

            result = agent.invoke("Problem", "Solution", "Context")

            assert result == "Analysis complete"
            agent.chain.invoke.assert_called_once()
