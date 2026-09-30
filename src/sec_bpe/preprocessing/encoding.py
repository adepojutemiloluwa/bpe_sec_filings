"""Character encoding normalization for SEC filings."""

import html
import re
from typing import Tuple


class EncodingNormalizer:
    """Normalize character encoding for SEC filings."""

    def __init__(self):
        """Initialize the encoding normalizer."""
        self.encoding_issues = []

    def normalize(self, text: str) -> str:
        """Normalize text encoding to UTF-8 with consistent Unicode representation.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text as UTF-8 string.
        """
        if not text:
            return text

        # Decode HTML entities
        text = self._decode_html_entities(text)

        # Normalize Unicode to NFC form (canonical composition)
        text = self._normalize_unicode(text)

        # Replace common problematic characters
        text = self._normalize_special_chars(text)

        return text

    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities to their Unicode equivalents.

        Args:
            text: Text with HTML entities.

        Returns:
            Text with entities decoded.
        """
        try:
            text = html.unescape(text)
        except Exception as e:
            self.encoding_issues.append(f"HTML entity decoding error: {e}")
        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFC form.

        Args:
            text: Input text.

        Returns:
            Normalized text.
        """
        import unicodedata
        return unicodedata.normalize("NFC", text)

    def _normalize_special_chars(self, text: str) -> str:
        """Replace problematic characters with standard equivalents.

        Args:
            text: Input text.

        Returns:
            Text with normalized special characters.
        """
        # Replace non-breaking spaces with regular spaces
        text = text.replace("\u00A0", " ")
        text = text.replace("\u200B", "")  # Zero-width space
        text = text.replace("\u200C", "")  # Zero-width non-joiner
        text = text.replace("\u200D", "")  # Zero-width joiner

        # Normalize smart quotes to regular quotes
        text = text.replace("\u201C", '"')  # Left double quote
        text = text.replace("\u201D", '"')  # Right double quote
        text = text.replace("\u2018", "'")  # Left single quote
        text = text.replace("\u2019", "'")  # Right single quote

        # Normalize dashes
        text = text.replace("\u2013", "-")  # En dash
        text = text.replace("\u2014", "--")  # Em dash
        text = text.replace("\u2015", "--")  # Horizontal bar

        return text

    def get_encoding_issues(self) -> list[str]:
        """Get list of encoding issues encountered.

        Returns:
            List of encoding issue messages.
        """
        return self.encoding_issues.copy()

    def reset(self) -> None:
        """Reset the encoding normalizer state."""
        self.encoding_issues.clear()
