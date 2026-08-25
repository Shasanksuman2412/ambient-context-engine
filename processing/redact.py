"""
Processing Layer — Privacy-Aware Redaction

Uses regular expressions to detect and redact Personally Identifiable Information
(PII) such as credit card numbers, email addresses, and phone numbers before
they are stored or embedded.
"""

import re
import logging

logger = logging.getLogger(__name__)

class PIIRedactor:
    def __init__(self):
        # Regular expressions for common PII
        self.patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b',
            "CREDIT_CARD": r'\b(?:\d[ -]*?){13,16}\b',
            # Simple phone number regex (can be improved based on locale)
            "PHONE_NUMBER": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        }
        
        # Compile regex for performance
        self.compiled_patterns = {
            name: re.compile(pattern) for name, pattern in self.patterns.items()
        }
        
        self._total_redactions = 0

    def redact(self, text: str) -> str:
        """
        Scans the text for PII and replaces it with [REDACTED_TYPE].
        """
        if not text:
            return text

        redacted_text = text
        for name, pattern in self.compiled_patterns.items():
            matches = pattern.findall(redacted_text)
            if matches:
                self._total_redactions += len(matches)
                # Replace with a labeled redaction tag
                redacted_text = pattern.sub(f'[REDACTED_{name}]', redacted_text)

        if redacted_text != text:
            logger.debug("PII detected and redacted from text.")

        return redacted_text

    @property
    def stats(self) -> dict:
        return {
            "total_redactions": self._total_redactions
        }

# ─── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    redactor = PIIRedactor()
    
    test_text = "Contact me at test.email@example.com or call 123-456-7890. My card is 4532 1234 5678 9012."
    print("Original:", test_text)
    print("Redacted:", redactor.redact(test_text))
    print("Stats:", redactor.stats)
