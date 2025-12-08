"""
Tests for Tools module.
"""
import pytest
import os

os.environ["GOOGLE_API_KEY"] = "test_api_key"

from src.agents.tools import (
    calculate,
    summarize,
    format_as_list,
    extract_keywords,
    get_tools,
    get_tool_descriptions
)


class TestCalculateTool:
    """Tests for calculate tool."""
    
    def test_basic_addition(self):
        """Test basic addition."""
        result = calculate.invoke({"expression": "2 + 2"})
        assert result == "4"
    
    def test_multiplication(self):
        """Test multiplication."""
        result = calculate.invoke({"expression": "3 * 4"})
        assert result == "12"
    
    def test_complex_expression(self):
        """Test complex expression."""
        result = calculate.invoke({"expression": "(10 + 5) * 2"})
        assert result == "30"
    
    def test_sqrt_function(self):
        """Test square root."""
        result = calculate.invoke({"expression": "sqrt(16)"})
        assert result == "4.0"
    
    def test_division_by_zero(self):
        """Test division by zero handling."""
        result = calculate.invoke({"expression": "1 / 0"})
        assert "error" in result.lower()
    
    def test_invalid_expression(self):
        """Test invalid expression handling."""
        result = calculate.invoke({"expression": "import os"})
        assert "error" in result.lower() or "invalid" in result.lower()


class TestSummarizeTool:
    """Tests for summarize tool."""
    
    def test_short_text_unchanged(self):
        """Test that short text is returned unchanged."""
        text = "This is short."
        result = summarize.invoke({"text": text})
        assert result == text
    
    def test_long_text_truncated(self):
        """Test that long text is summarized."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        result = summarize.invoke({"text": text, "max_sentences": 2})
        assert "First sentence" in result
        assert "Third sentence" not in result
    
    def test_empty_text(self):
        """Test empty text handling."""
        result = summarize.invoke({"text": ""})
        assert "no text" in result.lower()


class TestFormatAsListTool:
    """Tests for format_as_list tool."""
    
    def test_comma_separated(self):
        """Test comma-separated items."""
        result = format_as_list.invoke({"items": "apple, banana, cherry"})
        assert "• apple" in result
        assert "• banana" in result
        assert "• cherry" in result
    
    def test_newline_separated(self):
        """Test newline-separated items."""
        result = format_as_list.invoke({"items": "one\ntwo\nthree"})
        assert "• one" in result
        assert "• two" in result
    
    def test_ordered_list(self):
        """Test ordered list formatting."""
        result = format_as_list.invoke({"items": "first, second", "ordered": True})
        assert "1. first" in result
        assert "2. second" in result
    
    def test_empty_items(self):
        """Test empty items handling."""
        result = format_as_list.invoke({"items": ""})
        assert "no items" in result.lower()


class TestExtractKeywordsTool:
    """Tests for extract_keywords tool."""
    
    def test_keyword_extraction(self):
        """Test basic keyword extraction."""
        text = "Python programming language is great for machine learning and data science."
        result = extract_keywords.invoke({"text": text})
        
        # Should extract meaningful words, not stop words
        assert "the" not in result.lower()
        assert "is" not in result.lower()
    
    def test_max_keywords_limit(self):
        """Test max keywords limit."""
        text = "apple banana cherry date elderberry fig grape"
        result = extract_keywords.invoke({"text": text, "max_keywords": 3})
        keywords = result.split(", ")
        assert len(keywords) <= 3
    
    def test_empty_text(self):
        """Test empty text handling."""
        result = extract_keywords.invoke({"text": ""})
        assert "no text" in result.lower()


class TestToolRegistry:
    """Tests for tool registry functions."""
    
    def test_get_tools_returns_list(self):
        """Test that get_tools returns a list of tools."""
        tools = get_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 4  # calculate, summarize, format_as_list, extract_keywords
    
    def test_get_tool_descriptions(self):
        """Test that get_tool_descriptions returns formatted string."""
        descriptions = get_tool_descriptions()
        assert isinstance(descriptions, str)
        assert "calculate" in descriptions
        assert "summarize" in descriptions
