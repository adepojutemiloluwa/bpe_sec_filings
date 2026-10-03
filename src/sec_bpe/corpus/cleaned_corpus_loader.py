"""Load and validate Stage 2 cleaned corpus for Stage 3 preparation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional


class CleanedCorpusLoader:
    """Load and validate Stage 2 cleaned corpus documents."""

    def __init__(self, input_dir: str | Path):
        """Initialize the cleaned corpus loader.

        Args:
            input_dir: Directory containing Stage 2 cleaned output.
        """
        self.input_dir = Path(input_dir)
        self.logger = logging.getLogger(__name__)

    def load_corpus(self) -> List[Dict]:
        """Load and validate the cleaned corpus.

        Returns:
            List of document dictionaries with keys:
            - document_id: str
            - text: str
            - metadata: dict

        Raises:
            FileNotFoundError: If required files are missing.
        """
        corpus_path = self.input_dir / "corpus.txt"
        metadata_path = self.input_dir / "metadata.jsonl"
        report_path = self.input_dir / "cleaning_report.json"

        # Check required files exist
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load corpus text
        corpus_text = corpus_path.read_text(encoding="utf-8")

        # Load metadata
        metadata_records = self._load_metadata(metadata_path)

        # Load cleaning report for duplicate info
        duplicate_info = {}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            duplicate_info = report.get("duplicate_info", {})

        # Parse documents
        documents = self._parse_documents(corpus_text, metadata_records, duplicate_info)

        return documents

    def _load_metadata(self, metadata_path: Path) -> Dict[str, Dict]:
        """Load metadata JSONL into a dictionary.

        Args:
            metadata_path: Path to metadata.jsonl file.

        Returns:
            Dictionary mapping document_id to metadata record.
        """
        metadata = {}
        with metadata_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    doc_id = record.get("document_id")
                    if doc_id:
                        metadata[doc_id] = record
        return metadata

    def _parse_documents(
        self,
        corpus_text: str,
        metadata_records: Dict[str, Dict],
        duplicate_info: Dict,
    ) -> List[Dict]:
        """Parse documents from corpus text.

        Args:
            corpus_text: Raw corpus text with document markers.
            metadata_records: Metadata dictionary.
            duplicate_info: Duplicate information from cleaning report.

        Returns:
            List of document dictionaries.
        """
        documents = []

        # Split by DOCUMENT_START
        parts = corpus_text.split("<DOCUMENT_START>")

        for i, part in enumerate(parts):
            if not part.strip():
                continue

            # Extract document content before DOCUMENT_END
            if "<DOCUMENT_END>" in part:
                text, _ = part.split("<DOCUMENT_END>", 1)
                text = text.strip()
            else:
                # Malformed document - skip
                self.logger.warning(f"Document {i} missing <DOCUMENT_END> marker")
                continue

            # Generate document ID if not in metadata
            doc_id = f"doc_{i:06d}"

            # Get metadata
            metadata = metadata_records.get(doc_id, {})

            # Skip failed documents
            if metadata.get("status") == "failed":
                self.logger.info(f"Skipping failed document: {doc_id}")
                continue

            # Skip empty or very short documents
            if len(text) < 100:
                self.logger.warning(f"Skipping very short document: {doc_id} ({len(text)} chars)")
                continue

            # Skip duplicate documents (keep only the first occurrence)
            if doc_id in duplicate_info.get("duplicates", {}):
                original_id = duplicate_info["duplicates"][doc_id]
                self.logger.info(f"Skipping duplicate document: {doc_id} (original: {original_id})")
                continue

            documents.append({
                "document_id": doc_id,
                "text": text,
                "metadata": metadata,
            })

        return documents

    def validate_documents(self, documents: List[Dict]) -> Dict:
        """Validate loaded documents.

        Args:
            documents: List of document dictionaries.

        Returns:
            Validation report dictionary.
        """
        report = {
            "documents_loaded": len(documents),
            "documents_valid": 0,
            "documents_skipped": 0,
            "documents_failed": 0,
            "warnings": [],
            "errors": [],
        }

        for doc in documents:
            doc_id = doc["document_id"]
            text = doc["text"]
            metadata = doc["metadata"]

            # Check document ID
            if not doc_id:
                report["errors"].append("Document missing document_id")
                report["documents_failed"] += 1
                continue

            # Check text is not empty
            if not text:
                report["errors"].append(f"Document {doc_id} has empty text")
                report["documents_failed"] += 1
                continue

            # Check metadata consistency
            if metadata:
                if "cleaned_length" in metadata:
                    expected_length = metadata["cleaned_length"]
                    actual_length = len(text)
                    if abs(expected_length - actual_length) > 100:  # Allow some tolerance
                        report["warnings"].append(
                            f"Document {doc_id} length mismatch: "
                            f"metadata={expected_length}, actual={actual_length}"
                        )

            report["documents_valid"] += 1

        return report
