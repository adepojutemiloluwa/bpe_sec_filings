"""Whitespace normalization for SEC filings."""

import re


class WhitespaceNormalizer:
    """Normalize whitespace while preserving meaningful structure."""

    def __init__(self):
        """Initialize the whitespace normalizer."""
        pass

    def normalize(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph boundaries.

        Args:
            text: Input text.

        Returns:
            Text with normalized whitespace.
        """
        if not text:
            return text

        # Normalize line endings to \n
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing whitespace from lines
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

        # Normalize multiple spaces within lines to single space
        text = re.sub(r' +', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def remove_extraction_artifacts(self, text: str) -> str:
        """Remove common extraction artifacts like page numbers.

        Args:
            text: Input text.

        Returns:
            Text with extraction artifacts removed.
        """
        if not text:
            return text

        # Remove page number patterns (conservative)
        # Pattern: "Page X of Y" on its own line
        text = re.sub(r'\nPage \d+ of \d+\n', '\n', text, flags=re.IGNORECASE)

        # Remove standalone page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        return text

    def reset(self) -> None:
        """Reset the normalizer state."""
        pass
