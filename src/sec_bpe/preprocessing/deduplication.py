"""Duplicate detection for SEC filings."""

import hashlib
from typing import Dict, Optional


class DuplicateDetector:
    """Detect exact duplicate documents using hashing."""

    def __init__(self):
        """Initialize the duplicate detector."""
        self.document_hashes: Dict[str, str] = {}
        self.duplicates: Dict[str, str] = {}

    def compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of normalized text.

        Args:
            text: Input text to hash.

        Returns:
            Hexadecimal hash string.
        """
        # Normalize text for consistent hashing
        normalized = text.strip().lower()
        normalized = ' '.join(normalized.split())
        
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def check_duplicate(self, document_id: str, text: str) -> tuple[bool, Optional[str]]:
        """Check if a document is a duplicate of a previously seen document.

        Args:
            document_id: Unique identifier for the document.
            text: Document text content.

        Returns:
            Tuple of (is_duplicate, original_document_id).
        """
        doc_hash = self.compute_hash(text)

        if doc_hash in self.document_hashes:
            original_id = self.document_hashes[doc_hash]
            self.duplicates[document_id] = original_id
            return True, original_id

        self.document_hashes[doc_hash] = document_id
        return False, None

    def get_duplicate_info(self) -> Dict[str, str]:
        """Get information about detected duplicates.

        Returns:
            Dictionary mapping duplicate document IDs to original IDs.
        """
        return self.duplicates.copy()

    def get_stats(self) -> dict:
        """Get duplicate detection statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "total_documents": len(self.document_hashes) + len(self.duplicates),
            "unique_documents": len(self.document_hashes),
            "duplicate_documents": len(self.duplicates),
        }

    def reset(self) -> None:
        """Reset the detector state."""
        self.document_hashes.clear()
        self.duplicates.clear()
