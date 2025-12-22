"""
Guardrails module for input/output validation.

Provides safety checks and validation for agent inputs and outputs.
"""
import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class GuardrailStatus(Enum):
    """Status of guardrail validation."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    status: GuardrailStatus
    message: str
    original_input: str
    sanitized_input: Optional[str] = None
    violations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if the guardrail passed."""
        return self.status == GuardrailStatus.PASS

    @property
    def failed(self) -> bool:
        """Check if the guardrail failed."""
        return self.status == GuardrailStatus.FAIL


class InputGuardrail:
    """
    Input validation guardrail.
    
    Validates user inputs for safety and quality.
    """
    
    # Patterns that might indicate prompt injection attempts
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all)\s+instructions",
        r"disregard\s+(previous|all)\s+instructions",
        r"forget\s+(previous|all)\s+instructions",
        r"you\s+are\s+now\s+(?:a|an)\s+\w+",  # Role hijacking
        r"act\s+as\s+(?:a|an)\s+\w+",
        r"pretend\s+(?:you're|you\s+are)",
        r"system:\s*",  # System prompt injection
        r"\[INST\]",  # Common instruction markers
        r"<<SYS>>",
    ]
    
    # Forbidden content patterns
    FORBIDDEN_PATTERNS = [
        r"(?:create|make|generate)\s+(?:a\s+)?(?:bomb|weapon|explosive)",
        r"(?:how\s+to\s+)?(?:hack|break\s+into)",
    ]
    
    def __init__(
        self,
        max_length: int = 10000,
        min_length: int = 1,
        check_injection: bool = True,
        check_forbidden: bool = True
    ):
        """
        Initialize the input guardrail.
        
        Args:
            max_length: Maximum allowed input length.
            min_length: Minimum required input length.
            check_injection: Whether to check for prompt injection.
            check_forbidden: Whether to check for forbidden content.
        """
        self.max_length = max_length
        self.min_length = min_length
        self.check_injection = check_injection
        self.check_forbidden = check_forbidden
        
        # Compile patterns for efficiency
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        self._forbidden_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.FORBIDDEN_PATTERNS
        ]

    def validate(self, text: str) -> GuardrailResult:
        """
        Validate input text.
        
        Args:
            text: The input text to validate.
            
        Returns:
            GuardrailResult with validation status and details.
        """
        violations = []
        sanitized = text.strip()
        
        # Length checks
        if len(sanitized) < self.min_length:
            violations.append(f"Input too short (min: {self.min_length})")
        
        if len(sanitized) > self.max_length:
            violations.append(f"Input too long (max: {self.max_length})")
            sanitized = sanitized[:self.max_length]
        
        # Injection detection
        if self.check_injection:
            for pattern in self._injection_patterns:
                if pattern.search(sanitized):
                    violations.append(f"Potential prompt injection detected")
                    logger.warning(f"Injection pattern detected: {pattern.pattern}")
                    break
        
        # Forbidden content
        if self.check_forbidden:
            for pattern in self._forbidden_patterns:
                if pattern.search(sanitized):
                    violations.append("Forbidden content detected")
                    logger.warning(f"Forbidden pattern detected: {pattern.pattern}")
                    break
        
        # Determine status
        if violations:
            # Critical violations cause failure
            if any("injection" in v.lower() or "forbidden" in v.lower() for v in violations):
                status = GuardrailStatus.FAIL
                message = "Input rejected due to policy violation"
            else:
                status = GuardrailStatus.WARN
                message = "Input accepted with warnings"
        else:
            status = GuardrailStatus.PASS
            message = "Input validated successfully"
        
        return GuardrailResult(
            status=status,
            message=message,
            original_input=text,
            sanitized_input=sanitized,
            violations=violations
        )


class OutputGuardrail:
    """
    Output validation guardrail.
    
    Validates LLM outputs for safety and quality.
    """
    
    # Patterns indicating potential hallucination markers
    UNCERTAINTY_PHRASES = [
        "i'm not sure",
        "i don't know",
        "i cannot",
        "i can't",
        "as an ai",
        "i'm an ai",
        "i am an ai",
    ]
    
    # Patterns that should never appear in output
    FORBIDDEN_OUTPUT_PATTERNS = [
        r"(?:api[_\s]?key|password|secret)[:\s]+\S+",  # Credential leakage
        r"\b(?:ssn|social\s+security)\b.*\d{3}[-\s]?\d{2}[-\s]?\d{4}",  # SSN
    ]
    
    def __init__(
        self,
        max_length: int = 50000,
        check_credentials: bool = True,
        check_uncertainty: bool = True
    ):
        """
        Initialize the output guardrail.
        
        Args:
            max_length: Maximum allowed output length.
            check_credentials: Whether to check for credential leakage.
            check_uncertainty: Whether to flag uncertainty phrases.
        """
        self.max_length = max_length
        self.check_credentials = check_credentials
        self.check_uncertainty = check_uncertainty
        
        self._forbidden_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.FORBIDDEN_OUTPUT_PATTERNS
        ]

    def validate(self, text: str, question: Optional[str] = None) -> GuardrailResult:
        """
        Validate output text.
        
        Args:
            text: The output text to validate.
            question: Optional original question for context.
            
        Returns:
            GuardrailResult with validation status and details.
        """
        violations = []
        sanitized = text
        
        # Length check
        if len(text) > self.max_length:
            violations.append(f"Output too long (max: {self.max_length})")
            sanitized = text[:self.max_length] + "... [truncated]"
        
        # Credential leakage check
        if self.check_credentials:
            for pattern in self._forbidden_patterns:
                match = pattern.search(text)
                if match:
                    violations.append("Potential credential/PII leakage detected")
                    # Redact the match
                    sanitized = pattern.sub("[REDACTED]", sanitized)
                    logger.warning("Credential pattern found and redacted in output")
        
        # Uncertainty detection (warning only)
        if self.check_uncertainty:
            text_lower = text.lower()
            for phrase in self.UNCERTAINTY_PHRASES:
                if phrase in text_lower:
                    violations.append(f"Uncertainty detected: '{phrase}'")
                    break
        
        # Determine status
        has_critical = any(
            "credential" in v.lower() or "pii" in v.lower() 
            for v in violations
        )
        
        if has_critical:
            status = GuardrailStatus.FAIL
            message = "Output blocked due to safety concerns"
        elif violations:
            status = GuardrailStatus.WARN
            message = "Output passed with warnings"
        else:
            status = GuardrailStatus.PASS
            message = "Output validated successfully"
        
        return GuardrailResult(
            status=status,
            message=message,
            original_input=text,
            sanitized_input=sanitized,
            violations=violations
        )


def apply_guardrails(
    input_text: str,
    input_guardrail: Optional[InputGuardrail] = None,
    output_guardrail: Optional[OutputGuardrail] = None
) -> Tuple[bool, str, List[str]]:
    """
    Convenience function to apply input guardrails.
    
    Args:
        input_text: Text to validate.
        input_guardrail: Optional custom input guardrail.
        output_guardrail: Not used for input validation (for future use).
        
    Returns:
        Tuple of (passed, sanitized_text, violations).
    """
    guardrail = input_guardrail or InputGuardrail()
    result = guardrail.validate(input_text)
    return result.passed, result.sanitized_input or input_text, result.violations
