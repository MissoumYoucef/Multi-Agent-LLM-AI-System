"""
Tools module for agent tool calling.

Provides callable tools that agents can use to perform specific actions.
"""
import re
import math
import logging
from typing import List, Optional, Any
from langchain_core.tools import tool
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@tool
def search_documents(query: str, top_k: int = 3) -> str:
    """
    Search the knowledge base for relevant documents.
    
    Use this tool when you need to find information from the document store.
    
    Args:
        query: The search query to find relevant documents.
        top_k: Number of documents to retrieve (default: 3).
        
    Returns:
        A string containing the relevant document content.
    """
    # Note: This is a placeholder. The actual retriever is injected at runtime.
    logger.info(f"search_documents called with query: {query}")
    return f"[Search results for: {query}] - Retriever not connected. Connect via ReActAgent."


@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.
    
    Evaluates mathematical expressions safely. Supports basic operations,
    trigonometry, and common math functions.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "sin(3.14)", "sqrt(16)").
        
    Returns:
        The result of the calculation as a string.
    """
    # Safe math namespace
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }
    
    try:
        # Clean the expression
        cleaned = expression.strip()
        
        # Basic validation - only allow safe characters
        if not re.match(r'^[\d\s\+\-\*/\(\)\.\,a-z]+$', cleaned.lower()):
            return f"Error: Expression contains invalid characters"
        
        # Evaluate safely
        result = eval(cleaned, {"__builtins__": {}}, safe_dict)
        logger.info(f"Calculated: {expression} = {result}")
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        logger.warning(f"Calculation error: {e}")
        return f"Error: Could not evaluate '{expression}'. {str(e)}"


@tool
def summarize(text: str, max_sentences: int = 3) -> str:
    """
    Summarize a piece of text.
    
    Creates a brief summary by extracting key sentences.
    For LLM-based summarization, use the main agent.
    
    Args:
        text: The text to summarize.
        max_sentences: Maximum number of sentences in summary (default: 3).
        
    Returns:
        A summarized version of the text.
    """
    if not text or not text.strip():
        return "No text provided for summarization."
    
    # Simple extractive summarization using sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    if len(sentences) <= max_sentences:
        return text.strip()
    
    # Take first sentences as summary (simple heuristic)
    summary = ' '.join(sentences[:max_sentences])
    logger.info(f"Summarized {len(sentences)} sentences to {max_sentences}")
    return summary


@tool
def format_as_list(items: str, ordered: bool = False) -> str:
    """
    Format text items as a list.
    
    Converts comma-separated or newline-separated items into a formatted list.
    
    Args:
        items: Text containing items separated by commas or newlines.
        ordered: If True, creates a numbered list. Default is bullet list.
        
    Returns:
        A formatted list string.
    """
    # Split by newlines or commas
    if '\n' in items:
        item_list = [i.strip() for i in items.split('\n') if i.strip()]
    else:
        item_list = [i.strip() for i in items.split(',') if i.strip()]
    
    if not item_list:
        return "No items to format."
    
    if ordered:
        formatted = '\n'.join(f"{i+1}. {item}" for i, item in enumerate(item_list))
    else:
        formatted = '\n'.join(f"• {item}" for item in item_list)
    
    logger.info(f"Formatted {len(item_list)} items as {'ordered' if ordered else 'bullet'} list")
    return formatted


@tool
def extract_keywords(text: str, max_keywords: int = 5) -> str:
    """
    Extract key terms from text.
    
    Identifies important words based on frequency and position.
    
    Args:
        text: The text to extract keywords from.
        max_keywords: Maximum number of keywords to extract (default: 5).
        
    Returns:
        A comma-separated list of keywords.
    """
    if not text or not text.strip():
        return "No text provided for keyword extraction."
    
    # Simple keyword extraction based on word frequency
    # Exclude common stop words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
        'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'this', 'that', 'these', 'those', 'it', 'its',
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Count non-stop words
    word_counts = {}
    for word in words:
        if word not in stop_words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in sorted_words[:max_keywords]]
    
    logger.info(f"Extracted {len(keywords)} keywords from text")
    return ', '.join(keywords) if keywords else "No keywords found."


# Tool registry for easy access
AVAILABLE_TOOLS = [
    search_documents,
    calculate,
    summarize,
    format_as_list,
    extract_keywords,
]


def get_tools() -> List:
    """
    Get list of all available tools.
    
    Returns:
        List of tool objects that can be bound to an agent.
    """
    return AVAILABLE_TOOLS


def get_tool_descriptions() -> str:
    """
    Get formatted descriptions of all tools.
    
    Returns:
        A string with all tool names and descriptions.
    """
    descriptions = []
    for tool_obj in AVAILABLE_TOOLS:
        descriptions.append(f"- {tool_obj.name}: {tool_obj.description}")
    return '\n'.join(descriptions)
