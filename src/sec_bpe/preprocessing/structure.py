"""SEC filing structure preservation."""

import re


class StructurePreserver:
    """Preserve SEC filing structure during cleaning."""

    def __init__(self):
        """Initialize the structure preserver."""
        pass

    def preserve(self, text: str) -> str:
        """Preserve meaningful SEC filing structure.

        This ensures PART, ITEM, and section headings are preserved
        with appropriate spacing.

        Args:
            text: Input text.

        Returns:
            Text with structure preserved.
        """
        if not text:
            return text

        # Ensure PART headings have proper spacing
        text = self._preserve_part_headings(text)

        # Ensure ITEM headings have proper spacing
        text = self._preserve_item_headings(text)

        # Preserve section headings
        text = self._preserve_section_headings(text)

        return text

    def _preserve_part_headings(self, text: str) -> str:
        """Ensure PART headings have proper spacing.

        Args:
            text: Input text.

        Returns:
            Text with PART headings properly spaced.
        """
        # Add blank line before PART headings if missing
        text = re.sub(r'([^\n])\n(PART [IVX]+)', r'\1\n\nPART \2', text)
        
        # Ensure PART headings are on their own line
        text = re.sub(r'([^\n])PART ([IVX]+)', r'\1\nPART \2', text)
        
        return text

    def _preserve_item_headings(self, text: str) -> str:
        """Ensure ITEM headings have proper spacing.

        Args:
            text: Input text.

        Returns:
            Text with ITEM headings properly spaced.
        """
        # Add blank line before ITEM headings if missing
        text = re.sub(r'([^\n])\n(ITEM [0-9A-Z]+)', r'\1\n\nITEM \2', text)
        
        # Ensure ITEM headings are on their own line
        text = re.sub(r'([^\n])ITEM ([0-9A-Z]+)', r'\1\nITEM \2', text)
        
        return text

    def _preserve_section_headings(self, text: str) -> str:
        """Preserve section headings.

        Args:
            text: Input text.

        Returns:
            Text with section headings preserved.
        """
        # Detect common section heading patterns (all caps, followed by colon or period)
        # This is a simple heuristic - more sophisticated detection can be added later
        section_pattern = r'\n([A-Z][A-Z\s]{5,})([:\.])'
        
        # Ensure section headings have blank line before
        text = re.sub(section_pattern, r'\n\1\2', text)
        
        return text

    def reset(self) -> None:
        """Reset the preserver state."""
        pass
