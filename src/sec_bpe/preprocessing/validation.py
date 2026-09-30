"""Validation checks for cleaned corpus."""

import logging
from pathlib import Path
from typing import Dict, List


class CorpusValidator:
    """Validate cleaned corpus output."""

    def __init__(self):
        """Initialize the corpus validator."""
        self.logger = logging.getLogger(__name__)
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []

    def validate_corpus(
        self,
        corpus_path: Path,
        metadata_path: Path,
        report_path: Path,
    ) -> Dict:
        """Run all validation checks on the cleaned corpus.

        Args:
            corpus_path: Path to cleaned corpus file.
            metadata_path: Path to metadata JSONL file.
            report_path: Path to cleaning report JSON file.

        Returns:
            Dictionary with validation results.
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()

        self.logger.info("Starting corpus validation")

        # Check file existence
        self._check_file_existence(corpus_path, metadata_path, report_path)

        # Check document boundaries
        if corpus_path.exists():
            self._check_document_boundaries(corpus_path)

        # Check encoding
        if corpus_path.exists():
            self._check_encoding(corpus_path)

        # Check metadata consistency
        if metadata_path.exists():
            self._check_metadata_consistency(metadata_path, corpus_path)

        # Check for empty documents
        if corpus_path.exists():
            self._check_empty_documents(corpus_path)

        # Check financial text preservation
        if corpus_path.exists():
            self._check_financial_patterns(corpus_path)

        # Check report completeness
        if report_path.exists():
            self._check_report_completeness(report_path)

        results = {
            "is_valid": len(self.validation_errors) == 0,
            "errors": self.validation_errors,
            "warnings": self.validation_warnings,
        }

        if results["is_valid"]:
            self.logger.info("Corpus validation passed")
        else:
            self.logger.error(f"Corpus validation failed with {len(self.validation_errors)} errors")

        return results

    def _check_file_existence(
        self,
        corpus_path: Path,
        metadata_path: Path,
        report_path: Path,
    ) -> None:
        """Check that all required files exist.

        Args:
            corpus_path: Path to cleaned corpus file.
            metadata_path: Path to metadata file.
            report_path: Path to cleaning report file.
        """
        if not corpus_path.exists():
            self.validation_errors.append(f"Corpus file not found: {corpus_path}")

        if not metadata_path.exists():
            self.validation_errors.append(f"Metadata file not found: {metadata_path}")

        if not report_path.exists():
            self.validation_errors.append(f"Cleaning report not found: {report_path}")

    def _check_document_boundaries(self, corpus_path: Path) -> None:
        """Check that all documents have proper boundary markers.

        Args:
            corpus_path: Path to cleaned corpus file.
        """
        corpus_text = corpus_path.read_text(encoding="utf-8")

        # Count start and end markers
        start_count = corpus_text.count("<DOCUMENT_START>")
        end_count = corpus_text.count("<DOCUMENT_END>")

        if start_count == 0:
            self.validation_errors.append("No <DOCUMENT_START> markers found in corpus")

        if end_count == 0:
            self.validation_errors.append("No <DOCUMENT_END> markers found in corpus")

        if start_count != end_count:
            self.validation_errors.append(
                f"Document boundary mismatch: {start_count} start markers vs {end_count} end markers"
            )

    def _check_encoding(self, corpus_path: Path) -> None:
        """Check that corpus is valid UTF-8.

        Args:
            corpus_path: Path to cleaned corpus file.
        """
        try:
            corpus_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            self.validation_errors.append(f"Corpus encoding error: {e}")

    def _check_metadata_consistency(self, metadata_path: Path, corpus_path: Path) -> None:
        """Check that metadata is consistent with corpus.

        Args:
            metadata_path: Path to metadata file.
            corpus_path: Path to corpus file.
        """
        import json

        # Count documents in metadata
        metadata_count = 0
        with metadata_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata_count += 1

        # Count documents in corpus
        corpus_text = corpus_path.read_text(encoding="utf-8")
        corpus_count = corpus_text.count("<DOCUMENT_START>")

        if metadata_count != corpus_count:
            self.validation_warnings.append(
                f"Metadata count ({metadata_count}) differs from corpus document count ({corpus_count})"
            )

    def _check_empty_documents(self, corpus_path: Path) -> None:
        """Check for empty or extremely short documents.

        Args:
            corpus_path: Path to cleaned corpus file.
        """
        corpus_text = corpus_path.read_text(encoding="utf-8")
        documents = corpus_text.split("<DOCUMENT_START>")

        for i, doc in enumerate(documents):
            doc = doc.replace("<DOCUMENT_END>", "").strip()
            if len(doc) < 10:
                self.validation_warnings.append(f"Document {i} appears to be empty or very short")

    def _check_financial_patterns(self, corpus_path: Path) -> None:
        """Check for preservation of financial patterns.

        Args:
            corpus_path: Path to cleaned corpus file.
        """
        corpus_text = corpus_path.read_text(encoding="utf-8")

        # Check for common financial patterns
        patterns = {
            "percent_sign": "%",
            "dollar_sign": "$",
            "item_heading": "ITEM",
            "part_heading": "PART",
            "form_10k": "10-K",
            "form_10q": "10-Q",
        }

        for pattern_name, pattern in patterns.items():
            if pattern not in corpus_text:
                self.validation_warnings.append(
                    f"Financial pattern '{pattern}' not found in corpus"
                )

    def _check_report_completeness(self, report_path: Path) -> None:
        """Check that cleaning report contains required fields.

        Args:
            report_path: Path to cleaning report file.
        """
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))

        required_fields = [
            "statistics",
            "failed_documents",
            "duplicate_info",
        ]

        for field in required_fields:
            if field not in report:
                self.validation_errors.append(f"Missing required field in report: {field}")

        # Check statistics fields
        if "statistics" in report:
            required_stats = [
                "documents_processed",
                "documents_cleaned",
                "duplicates_detected",
            ]
            for stat in required_stats:
                if stat not in report["statistics"]:
                    self.validation_errors.append(f"Missing required statistic: {stat}")
