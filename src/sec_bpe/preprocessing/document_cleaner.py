"""Document cleaner pipeline orchestrator."""

import logging
from typing import Dict, Optional, Tuple

from sec_bpe.preprocessing.encoding import EncodingNormalizer
from sec_bpe.preprocessing.html_cleaner import HTMLCleaner
from sec_bpe.preprocessing.xbrl_cleaner import XBRLCleaner
from sec_bpe.preprocessing.table_parser import TableParser
from sec_bpe.preprocessing.structure import StructurePreserver
from sec_bpe.preprocessing.whitespace import WhitespaceNormalizer
from sec_bpe.preprocessing.deduplication import DuplicateDetector


class DocumentCleaner:
    """Orchestrate the document cleaning pipeline."""

    def __init__(self):
        """Initialize the document cleaner with all preprocessing components."""
        self.encoding_normalizer = EncodingNormalizer()
        self.html_cleaner = HTMLCleaner()
        self.xbrl_cleaner = XBRLCleaner()
        self.table_parser = TableParser()
        self.structure_preserver = StructurePreserver()
        self.whitespace_normalizer = WhitespaceNormalizer()
        self.duplicate_detector = DuplicateDetector()

        self.logger = logging.getLogger(__name__)

    def clean_document(self, document_id: str, text: str) -> Tuple[str, Dict]:
        """Clean a single document through the full pipeline.

        Args:
            document_id: Unique identifier for the document.
            text: Raw document text.

        Returns:
            Tuple of (cleaned_text, cleaning_metadata).
        """
        metadata = {
            "document_id": document_id,
            "original_length": len(text),
            "steps_completed": [],
            "errors": [],
        }

        try:
            # Step 1: Encoding normalization
            text = self.encoding_normalizer.normalize(text)
            metadata["steps_completed"].append("encoding_normalization")
            encoding_issues = self.encoding_normalizer.get_encoding_issues()
            if encoding_issues:
                metadata["encoding_issues"] = encoding_issues

            # Step 2: Table extraction (before HTML removal)
            text = self.table_parser.extract_tables(text)
            metadata["steps_completed"].append("table_extraction")
            metadata["table_count"] = self.table_parser.table_count

            # Step 3: HTML cleaning
            text = self.html_cleaner.clean(text)
            metadata["steps_completed"].append("html_cleaning")
            metadata["html_elements_removed"] = self.html_cleaner.elements_removed

            # Step 4: XBRL cleaning
            text = self.xbrl_cleaner.clean(text)
            metadata["steps_completed"].append("xbrl_cleaning")
            metadata["xbrl_elements_removed"] = self.xbrl_cleaner.xbrl_elements_removed

            # Step 5: Structure preservation
            text = self.structure_preserver.preserve(text)
            metadata["steps_completed"].append("structure_preservation")

            # Step 6: Whitespace normalization
            text = self.whitespace_normalizer.normalize(text)
            metadata["steps_completed"].append("whitespace_normalization")

            # Step 7: Remove extraction artifacts
            text = self.whitespace_normalizer.remove_extraction_artifacts(text)
            metadata["steps_completed"].append("artifact_removal")

            metadata["cleaned_length"] = len(text)
            metadata["status"] = "success"

            self.logger.info(f"Successfully cleaned document {document_id}")

        except Exception as e:
            metadata["status"] = "failed"
            metadata["errors"].append(str(e))
            self.logger.error(f"Failed to clean document {document_id}: {e}")
            # Return original text on failure
            text = ""

        return text, metadata

    def check_duplicate(self, document_id: str, text: str) -> Tuple[bool, Optional[str]]:
        """Check if a document is a duplicate.

        Args:
            document_id: Unique identifier for the document.
            text: Document text content.

        Returns:
            Tuple of (is_duplicate, original_document_id).
        """
        return self.duplicate_detector.check_duplicate(document_id, text)

    def get_aggregate_stats(self) -> Dict:
        """Get aggregate statistics from all components.

        Returns:
            Dictionary with aggregate statistics.
        """
        return {
            "encoding_issues": len(self.encoding_normalizer.get_encoding_issues()),
            "html_elements_removed": self.html_cleaner.elements_removed,
            "xbrl_elements_removed": self.xbrl_cleaner.xbrl_elements_removed,
            "tables_processed": self.table_parser.table_count,
            "duplicate_stats": self.duplicate_detector.get_stats(),
        }

    def reset(self) -> None:
        """Reset all component states."""
        self.encoding_normalizer.reset()
        self.html_cleaner.reset()
        self.xbrl_cleaner.reset()
        self.table_parser.reset()
        self.structure_preserver.reset()
        self.whitespace_normalizer.reset()
        self.duplicate_detector.reset()
