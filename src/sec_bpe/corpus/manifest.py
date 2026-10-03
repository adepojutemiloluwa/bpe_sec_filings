"""Generate reproducible split manifests."""

import json
from pathlib import Path
from typing import Dict, List

from sec_bpe.corpus.config import CorpusPreparationConfig


class ManifestGenerator:
    """Generate reproducible split manifests."""

    def __init__(self, config: CorpusPreparationConfig):
        """Initialize the manifest generator.

        Args:
            config: Corpus preparation configuration.
        """
        self.config = config

    def generate(
        self,
        splits: Dict[str, List[Dict]],
        company_assignments: Dict[str, str] = None,
    ) -> Dict:
        """Generate split manifest.

        Args:
            splits: Dictionary with train/validation/test document lists.
            company_assignments: Optional company-to-split mapping.

        Returns:
            Manifest dictionary.
        """
        manifest = {
            "strategy": self.config.split_strategy,
            "seed": self.config.seed,
            "train": [],
            "validation": [],
            "test": [],
        }

        # Add company assignments if available
        if company_assignments:
            manifest["company_assignments"] = company_assignments

        # Add document IDs to each split
        for split_name, documents in splits.items():
            manifest[split_name] = sorted([doc["document_id"] for doc in documents])

        return manifest

    def save_manifest(self, manifest: Dict, output_path: Path) -> None:
        """Save manifest to JSON file.

        Args:
            manifest: Manifest dictionary.
            output_path: Path to save manifest.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def load_manifest(self, manifest_path: Path) -> Dict:
        """Load manifest from JSON file.

        Args:
            manifest_path: Path to manifest file.

        Returns:
            Manifest dictionary.
        """
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def validate_manifest(
        self,
        manifest: Dict,
        documents: List[Dict],
    ) -> Dict:
        """Validate manifest against documents.

        Args:
            manifest: Manifest dictionary.
            documents: List of document dictionaries.

        Returns:
            Validation report.
        """
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }

        # Get all document IDs
        all_doc_ids = {doc["document_id"] for doc in documents}

        # Get manifest document IDs
        manifest_doc_ids = set()
        for split_name in ["train", "validation", "test"]:
            manifest_doc_ids.update(manifest.get(split_name, []))

        # Check that all documents are in manifest
        missing_docs = all_doc_ids - manifest_doc_ids
        if missing_docs:
            report["errors"].append(
                f"Documents missing from manifest: {len(missing_docs)}"
            )
            report["is_valid"] = False

        # Check that manifest has no extra documents
        extra_docs = manifest_doc_ids - all_doc_ids
        if extra_docs:
            report["warnings"].append(
                f"Manifest has extra documents: {len(extra_docs)}"
            )

        # Check for duplicates across splits
        train_set = set(manifest.get("train", []))
        val_set = set(manifest.get("validation", []))
        test_set = set(manifest.get("test", []))

        train_val_overlap = train_set & val_set
        if train_val_overlap:
            report["errors"].append(
                f"Train/validation overlap: {len(train_val_overlap)} documents"
            )
            report["is_valid"] = False

        train_test_overlap = train_set & test_set
        if train_test_overlap:
            report["errors"].append(
                f"Train/test overlap: {len(train_test_overlap)} documents"
            )
            report["is_valid"] = False

        val_test_overlap = val_set & test_set
        if val_test_overlap:
            report["errors"].append(
                f"Validation/test overlap: {len(val_test_overlap)} documents"
            )
            report["is_valid"] = False

        return report
