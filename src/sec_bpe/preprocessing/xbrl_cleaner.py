"""XBRL handling for SEC filings."""

import re
from typing import Tuple


class XBRLCleaner:
    """Clean XBRL markup from SEC filings while preserving human-readable content."""

    def __init__(self):
        """Initialize the XBRL cleaner."""
        self.xbrl_elements_removed = 0

    def clean(self, text: str) -> str:
        """Remove XBRL markup while preserving human-readable content.

        Args:
            text: Input text with XBRL markup.

        Returns:
            Text with XBRL markup removed.
        """
        if not text:
            return text

        # Remove XBRL context definitions
        text = self._remove_context(text)

        # Remove XBRL unit definitions
        text = self._remove_units(text)

        # Remove XBRL schema references
        text = self._remove_schema_refs(text)

        # Remove XBRL namespace prefixes from tags (but preserve content)
        text = self._remove_xbrl_tags(text)

        return text

    def _remove_context(self, text: str) -> str:
        """Remove XBRL context elements.

        Args:
            text: Input text.

        Returns:
            Text with context removed.
        """
        pattern = r'<xbrli:context[^>]*>.*?</xbrli:context>'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        self.xbrl_elements_removed += len(matches)
        return re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    def _remove_units(self, text: str) -> str:
        """Remove XBRL unit elements.

        Args:
            text: Input text.

        Returns:
            Text with units removed.
        """
        pattern = r'<xbrli:unit[^>]*>.*?</xbrli:unit>'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        self.xbrl_elements_removed += len(matches)
        return re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    def _remove_schema_refs(self, text: str) -> str:
        """Remove XBRL schema references.

        Args:
            text: Input text.

        Returns:
            Text with schema refs removed.
        """
        pattern = r'<link:[^>]+xbrl[^>]*>'
        matches = re.findall(pattern, text, re.IGNORECASE)
        self.xbrl_elements_removed += len(matches)
        return re.sub(pattern, '', text, flags=re.IGNORECASE)

    def _remove_xbrl_tags(self, text: str) -> str:
        """Remove XBRL-specific tags while preserving content.

        This removes tags with XBRL namespaces but keeps the text content.

        Args:
            text: Input text.

        Returns:
            Text with XBRL tags removed.
        """
        # Remove tags with common XBRL namespaces
        namespaces = ['us-gaap:', 'dei:', 'xbrli:', 'xbrl:']
        
        for ns in namespaces:
            # Remove opening tags
            pattern = f'<{ns}[^>]+>'
            matches = re.findall(pattern, text, re.IGNORECASE)
            self.xbrl_elements_removed += len(matches)
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            # Remove closing tags
            pattern = f'</{ns}[^>]*>'
            matches = re.findall(pattern, text, re.IGNORECASE)
            self.xbrl_elements_removed += len(matches)
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text

    def get_stats(self) -> dict:
        """Get cleaning statistics.

        Returns:
            Dictionary with cleaning statistics.
        """
        return {
            "xbrl_elements_removed": self.xbrl_elements_removed,
        }

    def reset(self) -> None:
        """Reset the cleaner state."""
        self.xbrl_elements_removed = 0
