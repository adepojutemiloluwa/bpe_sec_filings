"""Table parsing for SEC filings."""

import re
from typing import List, Tuple


class TableParser:
    """Parse and extract table content from SEC filings."""

    def __init__(self):
        """Initialize the table parser."""
        self.table_count = 0

    def extract_tables(self, text: str) -> str:
        """Extract and serialize table content.

        Args:
            text: Input text with HTML tables.

        Returns:
            Text with tables converted to linear format.
        """
        if not text:
            return text

        # Find and replace all table elements
        table_pattern = r'<table[^>]*>(.*?)</table>'
        
        def replace_table(match):
            """Replace a table match with serialized content."""
            table_content = match.group(1)
            serialized = self._serialize_table(table_content)
            self.table_count += 1
            return serialized
        
        text = re.sub(table_pattern, replace_table, text, flags=re.DOTALL | re.IGNORECASE)

        return text

    def _serialize_table(self, table_html: str) -> str:
        """Serialize a table to linear text format.

        Args:
            table_html: HTML table content.

        Returns:
            Linear text representation of the table.
        """
        # Extract rows
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        rows = re.findall(row_pattern, table_html, re.DOTALL | re.IGNORECASE)

        serialized_rows = []
        for row in rows:
            # Extract cells
            cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
            cells = re.findall(cell_pattern, row, re.DOTALL | re.IGNORECASE)
            
            # Clean cell text
            cleaned_cells = [self._clean_cell_text(cell) for cell in cells]
            
            if cleaned_cells:
                serialized_rows.append(' '.join(cleaned_cells))

        if serialized_rows:
            return '\n'.join(serialized_rows) + '\n'
        return ''

    def _clean_cell_text(self, cell_text: str) -> str:
        """Clean text within a table cell.

        Args:
            cell_text: Raw cell text.

        Returns:
            Cleaned cell text.
        """
        # Remove any remaining HTML tags
        cell_text = re.sub(r'<[^>]+>', '', cell_text)
        
        # Normalize whitespace
        cell_text = re.sub(r'\s+', ' ', cell_text).strip()
        
        return cell_text

    def get_stats(self) -> dict:
        """Get parsing statistics.

        Returns:
            Dictionary with parsing statistics.
        """
        return {
            "table_count": self.table_count,
        }

    def reset(self) -> None:
        """Reset the parser state."""
        self.table_count = 0
