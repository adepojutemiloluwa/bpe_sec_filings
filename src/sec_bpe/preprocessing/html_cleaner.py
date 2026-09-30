"""HTML/XML removal for SEC filings."""

from typing import Tuple


class HTMLCleaner:
    """Clean HTML/XML markup from SEC filings while preserving text."""

    def __init__(self):
        """Initialize the HTML cleaner."""
        self.elements_removed = 0

    def clean(self, text: str) -> str:
        """Remove HTML/XML markup while preserving textual content.

        Args:
            text: Input text with HTML/XML markup.

        Returns:
            Text with markup removed.
        """
        if not text:
            return text

        # Remove script and style tags with their content
        text = self._remove_script_style(text)

        # Remove HTML comments
        text = self._remove_comments(text)

        # Remove all HTML tags but preserve content
        text = self._remove_tags(text)

        return text

    def _remove_script_style(self, text: str) -> str:
        """Remove script and style tags and their content.

        Args:
            text: Input text.

        Returns:
            Text with script/style removed.
        """
        import re
        pattern = r'<(script|style)[^>]*>.*?</\1>'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        self.elements_removed += len(matches)
        return re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    def _remove_comments(self, text: str) -> str:
        """Remove HTML comments.

        Args:
            text: Input text.

        Returns:
            Text with comments removed.
        """
        import re
        pattern = r'<!--.*?-->'
        matches = re.findall(pattern, text, re.DOTALL)
        self.elements_removed += len(matches)
        return re.sub(pattern, '', text, flags=re.DOTALL)

    def _remove_tags(self, text: str) -> str:
        """Remove HTML tags while preserving content.

        Args:
            text: Input text.

        Returns:
            Text with tags removed.
        """
        import re
        # Remove self-closing tags
        self_closing = r'<[^>]+/>'
        matches = re.findall(self_closing, text)
        self.elements_removed += len(matches)
        text = re.sub(self_closing, '', text)

        # Remove opening and closing tags
        tag_pattern = r'<[^>]+>'
        matches = re.findall(tag_pattern, text)
        self.elements_removed += len(matches)
        text = re.sub(tag_pattern, '', text)

        return text

    def get_stats(self) -> dict:
        """Get cleaning statistics.

        Returns:
            Dictionary with cleaning statistics.
        """
        return {
            "elements_removed": self.elements_removed,
        }

    def reset(self) -> None:
        """Reset the cleaner state."""
        self.elements_removed = 0
