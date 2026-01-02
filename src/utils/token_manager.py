"""
Token Manager module.

Provides token counting, context truncation, and prompt compression
for efficient LLM usage.
"""
import logging
import re
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Manages token counting and reduction for LLM prompts.

    Uses tiktoken for accurate token counting when available,
    with fallback to character-based estimation.
    """

    # Approximate tokens per character for estimation
    CHARS_PER_TOKEN = 4

    # Model token limits
    MODEL_LIMITS = {
        "gemini-pro": 32000,
        "gemini-1.5-pro": 128000,
        "gemini-flash": 32000,
        "gemini-1.5-flash": 128000,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-3.5-turbo": 4096,
    }

    def __init__(self, default_model: str = "gemini-pro"):
        """
        Initialize token manager.

        Args:
            default_model: Default model for token limits.
        """
        self.default_model = default_model
        self._encoder = None
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        """Initialize tokenizer with fallback."""
        try:
            import tiktoken
            # Use cl100k_base as a reasonable approximation for most models
            self._encoder = tiktoken.get_encoding("cl100k_base")
            logger.info("Token manager initialized with tiktoken")
        except ImportError:
            logger.warning("tiktoken not available, using character-based estimation")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for.
            model: Optional model name (unused, for API compatibility).

        Returns:
            Token count.
        """
        if not text:
            return 0

        if self._encoder:
            return len(self._encoder.encode(text))

        # Fallback: estimate based on characters
        return len(text) // self.CHARS_PER_TOKEN

    def get_model_limit(self, model: Optional[str] = None) -> int:
        """Get token limit for a model."""
        model = model or self.default_model
        return self.MODEL_LIMITS.get(model, 32000)

    def truncate_to_limit(
        self,
        text: str,
        max_tokens: int,
        truncation_strategy: str = "end",
        preserve_sentences: bool = True
    ) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate.
            max_tokens: Maximum tokens allowed.
            truncation_strategy: 'start', 'end', or 'middle'.
            preserve_sentences: Try to break at sentence boundaries.

        Returns:
            Truncated text.
        """
        current_tokens = self.count_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Calculate target character length
        ratio = max_tokens / current_tokens
        target_chars = int(len(text) * ratio * 0.95)  # 5% buffer

        if truncation_strategy == "end":
            truncated = text[:target_chars]
        elif truncation_strategy == "start":
            truncated = text[-target_chars:]
        else:  # middle
            half = target_chars // 2
            truncated = text[:half] + "\n...[truncated]...\n" + text[-half:]

        # Preserve sentence boundaries if requested
        if preserve_sentences and truncation_strategy == "end":
            # Find last sentence boundary
            last_period = truncated.rfind('. ')
            if last_period > len(truncated) * 0.5:
                truncated = truncated[:last_period + 1]

        logger.debug(f"Truncated from {current_tokens} to ~{self.count_tokens(truncated)} tokens")
        return truncated

    def compress_context(
        self,
        context: str,
        target_ratio: float = 0.5,
        preserve_key_info: bool = True
    ) -> str:
        """
        Compress context to reduce token usage.

        Args:
            context: Context string to compress.
            target_ratio: Target size as ratio of original.
            preserve_key_info: Try to preserve key information.

        Returns:
            Compressed context.
        """
        if not context:
            return context

        original_tokens = self.count_tokens(context)
        target_tokens = int(original_tokens * target_ratio)

        # Strategy 1: Remove redundant whitespace
        compressed = re.sub(r'\s+', ' ', context)
        compressed = re.sub(r'\n\s*\n', '\n', compressed)

        # Strategy 2: Remove common filler phrases
        filler_patterns = [
            r'\b(basically|essentially|actually|literally|honestly|frankly)\b',
            r'\b(you know|I mean|kind of|sort of)\b',
            r'\b(in order to)\b',
            r'\b(the fact that)\b',
        ]
        for pattern in filler_patterns:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)

        # Strategy 3: Truncate if still too long
        current = self.count_tokens(compressed)
        if current > target_tokens:
            compressed = self.truncate_to_limit(
                compressed,
                target_tokens,
                truncation_strategy="end",
                preserve_sentences=True
            )

        logger.debug(f"Compressed context: {original_tokens} -> {self.count_tokens(compressed)} tokens")
        return compressed.strip()

    def split_into_chunks(
        self,
        text: str,
        chunk_size: int = 4000,
        overlap: int = 200
    ) -> List[str]:
        """
        Split text into token-based chunks.

        Args:
            text: Text to split.
            chunk_size: Maximum tokens per chunk.
            overlap: Token overlap between chunks.

        Returns:
            List of text chunks.
        """
        if self.count_tokens(text) <= chunk_size:
            return [text]

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep overlap from end of previous chunk
                    overlap_sentences = []
                    overlap_tokens = 0
                    for s in reversed(current_chunk):
                        s_tokens = self.count_tokens(s)
                        if overlap_tokens + s_tokens <= overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """
        Estimate API cost for token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.

        Returns:
            Estimated cost in USD.
        """
        model = model or self.default_model

        # Approximate pricing (USD per 1K tokens)
        pricing = {
            "gemini-pro": {"input": 0.00025, "output": 0.0005},
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
            "gemini-flash": {"input": 0.000075, "output": 0.0003},
            "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }

        rates = pricing.get(model, {"input": 0.001, "output": 0.002})

        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]

        return input_cost + output_cost

    def get_stats(self, text: str) -> Dict[str, Any]:
        """Get token statistics for text."""
        tokens = self.count_tokens(text)
        return {
            "tokens": tokens,
            "characters": len(text),
            "chars_per_token": len(text) / tokens if tokens > 0 else 0,
            "estimated_cost_1k_output": self.estimate_cost(tokens, 1000)
        }


# Factory function
def create_token_manager(model: str = "gemini-pro") -> TokenManager:
    """Create a token manager with specified default model."""
    return TokenManager(default_model=model)
