"""
Unit tests for EvaluationMetrics class.

Tests functional correctness, lexical exactness, and ROUGE-L metrics
without requiring actual LLM API calls.
"""
import pytest
import os
from unittest.mock import patch, MagicMock

# Set env var before importing
os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestFunctionalCorrectness:
    """Tests for the functional_correctness metric."""
    
    def test_all_keywords_present(self):
        """All keywords found should return 1.0."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            prediction = "Machine learning and artificial intelligence are related fields."
            keywords = ["machine", "learning", "intelligence"]
            
            score = metrics.functional_correctness(prediction, keywords)
            assert score == 1.0
    
    def test_no_keywords_present(self):
        """No keywords found should return 0.0."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            prediction = "The weather is nice today."
            keywords = ["machine", "learning", "ai"]
            
            score = metrics.functional_correctness(prediction, keywords)
            assert score == 0.0
    
    def test_partial_keywords(self):
        """Some keywords found should return proportional score."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            prediction = "Machine learning is fascinating."
            keywords = ["machine", "learning", "neural", "networks"]
            
            score = metrics.functional_correctness(prediction, keywords)
            assert score == 0.5  # 2 out of 4
    
    def test_empty_keywords(self):
        """Empty keywords list should return 1.0."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            prediction = "Any text here."
            keywords = []
            
            score = metrics.functional_correctness(prediction, keywords)
            assert score == 1.0
    
    def test_empty_prediction(self):
        """Empty prediction should return 0.0."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            prediction = ""
            keywords = ["machine", "learning"]
            
            score = metrics.functional_correctness(prediction, keywords)
            assert score == 0.0
    
    def test_case_insensitive(self):
        """Keyword matching should be case insensitive."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            prediction = "MACHINE LEARNING is great!"
            keywords = ["machine", "Learning"]
            
            score = metrics.functional_correctness(prediction, keywords)
            assert score == 1.0


class TestLexicalExactness:
    """Tests for the lexical_exactness metric."""
    
    def test_identical_strings(self):
        """Identical strings should return 1.0."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            text = "This is a test."
            score = metrics.lexical_exactness(text, text)
            assert score == 1.0
    
    def test_completely_different(self):
        """Completely different strings should return low score."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            prediction = "abc"
            reference = "xyz"
            
            score = metrics.lexical_exactness(prediction, reference)
            assert score < 0.5
    
    def test_similar_strings(self):
        """Similar strings should return high score."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            prediction = "Machine learning is great"
            reference = "Machine learning is good"
            
            score = metrics.lexical_exactness(prediction, reference)
            assert score > 0.8
    
    def test_empty_strings(self):
        """Empty strings should return 0.0."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            score = metrics.lexical_exactness("", "reference")
            assert score == 0.0
            
            score = metrics.lexical_exactness("prediction", "")
            assert score == 0.0


class TestRougeLScore:
    """Tests for the rouge_l_score metric."""
    
    def test_identical_strings(self):
        """Identical strings should return 1.0."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            text = "The quick brown fox jumps"
            score = metrics.rouge_l_score(text, text)
            assert score == 1.0
    
    def test_no_overlap(self):
        """No overlapping words should return 0.0."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            prediction = "aaa bbb ccc"
            reference = "xxx yyy zzz"
            
            score = metrics.rouge_l_score(prediction, reference)
            assert score == 0.0
    
    def test_partial_overlap(self):
        """Partial overlap should return score between 0 and 1."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            prediction = "the cat sat on mat"
            reference = "the cat is on the mat"
            
            score = metrics.rouge_l_score(prediction, reference)
            assert 0 < score < 1
    
    def test_empty_strings(self):
        """Empty strings should return 0.0."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            score = metrics.rouge_l_score("", "reference text")
            assert score == 0.0
            
            score = metrics.rouge_l_score("prediction text", "")
            assert score == 0.0


class TestLCSLength:
    """Tests for the _lcs_length helper method."""
    
    def test_identical_sequences(self):
        """Identical sequences should return full length."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            seq = ["a", "b", "c"]
            lcs = metrics._lcs_length(seq, seq)
            assert lcs == 3
    
    def test_no_common_elements(self):
        """No common elements should return 0."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            lcs = metrics._lcs_length(["a", "b"], ["x", "y"])
            assert lcs == 0
    
    def test_subsequence(self):
        """Should find longest common subsequence."""
        from src.evaluation.metrics import EvaluationMetrics
        
        with patch('src.evaluation.metrics.ChatGoogleGenerativeAI'):
            metrics = EvaluationMetrics()
            
            lcs = metrics._lcs_length(
                ["a", "b", "c", "d"],
                ["b", "c"]
            )
            assert lcs == 2
