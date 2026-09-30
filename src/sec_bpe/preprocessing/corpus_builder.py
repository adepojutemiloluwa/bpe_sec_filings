"""Cleaned corpus builder for Stage 2."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from sec_bpe.preprocessing.document_cleaner import DocumentCleaner


class CleanedCorpusBuilder:
    """Build cleaned corpus from raw SEC filings."""

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ):
        """Initialize the cleaned corpus builder.

        Args:
            input_dir: Directory containing raw corpus files.
            output_dir: Directory where cleaned corpus will be written.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.document_cleaner = DocumentCleaner()
        self.logger = logging.getLogger(__name__)

        # Statistics tracking
        self.stats = {
            "documents_processed": 0,
            "documents_cleaned": 0,
            "documents_failed": 0,
            "duplicates_detected": 0,
            "total_raw_characters": 0,
            "total_cleaned_characters": 0,
            "html_elements_removed": 0,
            "xbrl_elements_removed": 0,
            "table_count": 0,
            "encoding_issues": 0,
        }

        # Metadata tracking
        self.metadata_list: List[Dict] = []
        self.failed_documents: List[Dict] = []

    def build(self) -> Dict:
        """Build the cleaned corpus from raw documents.

        Returns:
            Dictionary with build statistics.
        """
        self.logger.info(f"Starting cleaned corpus build from {self.input_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Read raw corpus
        raw_corpus_path = self.input_dir / "corpus.txt"
        if not raw_corpus_path.exists():
            raise FileNotFoundError(f"Raw corpus not found: {raw_corpus_path}")

        raw_corpus = raw_corpus_path.read_text(encoding="utf-8", errors="replace")

        # Split into individual documents
        documents = self._split_documents(raw_corpus)

        self.logger.info(f"Found {len(documents)} documents in raw corpus")

        # Process each document
        cleaned_documents = []
        for i, doc_text in enumerate(documents):
            document_id = f"doc_{i:06d}"
            
            # Clean document
            cleaned_text, metadata = self.document_cleaner.clean_document(
                document_id, doc_text
            )

            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["total_raw_characters"] += metadata.get("original_length", 0)
            self.stats["total_cleaned_characters"] += metadata.get("cleaned_length", 0)
            self.stats["html_elements_removed"] += metadata.get("html_elements_removed", 0)
            self.stats["xbrl_elements_removed"] += metadata.get("xbrl_elements_removed", 0)
            self.stats["table_count"] += metadata.get("table_count", 0)
            self.stats["encoding_issues"] += len(metadata.get("encoding_issues", []))

            if metadata["status"] == "success":
                self.stats["documents_cleaned"] += 1
                
                # Check for duplicates
                is_duplicate, original_id = self.document_cleaner.check_duplicate(
                    document_id, cleaned_text
                )
                if is_duplicate:
                    self.stats["duplicates_detected"] += 1
                    metadata["is_duplicate"] = True
                    metadata["duplicate_of"] = original_id
                    self.logger.info(f"Duplicate detected: {document_id} of {original_id}")
                else:
                    metadata["is_duplicate"] = False
                    cleaned_documents.append((document_id, cleaned_text))

                self.metadata_list.append(metadata)
            else:
                self.stats["documents_failed"] += 1
                self.failed_documents.append(metadata)
                self.logger.error(f"Failed to clean document {document_id}")

        # Write cleaned corpus
        self._write_cleaned_corpus(cleaned_documents)

        # Write metadata
        self._write_metadata()

        # Write cleaning report
        self._write_cleaning_report()

        self.logger.info("Cleaned corpus build completed")

        return self.stats

    def _split_documents(self, corpus_text: str) -> List[str]:
        """Split corpus text into individual documents.

        Args:
            corpus_text: Raw corpus text with document markers.

        Returns:
            List of individual document texts.
        """
        documents = []
        
        # Split by document markers
        parts = corpus_text.split("<document_start>")
        
        for part in parts:
            # Remove end marker and strip
            doc = part.replace("<document_end>", "").strip()
            if doc:
                documents.append(doc)
        
        return documents

    def _write_cleaned_corpus(self, documents: List[tuple[str, str]]) -> None:
        """Write cleaned documents to corpus file.

        Args:
            documents: List of (document_id, cleaned_text) tuples.
        """
        corpus_path = self.output_dir / "corpus.txt"
        
        with corpus_path.open("w", encoding="utf-8") as f:
            for i, (doc_id, doc_text) in enumerate(documents):
                if i > 0:
                    f.write("\n\n")
                f.write("<DOCUMENT_START>\n")
                f.write(doc_text)
                f.write("\n<DOCUMENT_END>")

        self.logger.info(f"Wrote cleaned corpus to {corpus_path}")

    def _write_metadata(self) -> None:
        """Write metadata to JSONL file."""
        metadata_path = self.output_dir / "metadata.jsonl"
        
        with metadata_path.open("w", encoding="utf-8") as f:
            for metadata in self.metadata_list:
                f.write(json.dumps(metadata) + "\n")

        self.logger.info(f"Wrote metadata to {metadata_path}")

    def _write_cleaning_report(self) -> None:
        """Write cleaning report to JSON file."""
        report_path = self.output_dir / "cleaning_report.json"
        
        report = {
            "statistics": self.stats,
            "failed_documents": self.failed_documents,
            "duplicate_info": self.document_cleaner.duplicate_detector.get_duplicate_info(),
        }
        
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Wrote cleaning report to {report_path}")

    def reset(self) -> None:
        """Reset the builder state."""
        self.document_cleaner.reset()
        self.stats = {
            "documents_processed": 0,
            "documents_cleaned": 0,
            "documents_failed": 0,
            "duplicates_detected": 0,
            "total_raw_characters": 0,
            "total_cleaned_characters": 0,
            "html_elements_removed": 0,
            "xbrl_elements_removed": 0,
            "table_count": 0,
            "encoding_issues": 0,
        }
        self.metadata_list.clear()
        self.failed_documents.clear()
