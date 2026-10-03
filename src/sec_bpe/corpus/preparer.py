"""Stage 3 orchestrator for BPE corpus preparation."""

import json
import logging
from pathlib import Path
from typing import Dict, List

from sec_bpe.corpus.config import CorpusPreparationConfig
from sec_bpe.corpus.cleaned_corpus_loader import CleanedCorpusLoader
from sec_bpe.corpus.splitter import CorpusSplitter
from sec_bpe.corpus.statistics import CorpusStatistics
from sec_bpe.corpus.manifest import ManifestGenerator


class CorpusPreparer:
    """Orchestrate Stage 3 corpus preparation."""

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        config: CorpusPreparationConfig = None,
    ):
        """Initialize the corpus preparer.

        Args:
            input_dir: Directory containing Stage 2 cleaned output.
            output_dir: Directory for Stage 3 prepared output.
            config: Corpus preparation configuration.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or CorpusPreparationConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.loader = CleanedCorpusLoader(self.input_dir)
        self.splitter = CorpusSplitter(self.config)
        self.statistics = CorpusStatistics()
        self.manifest_generator = ManifestGenerator(self.config)

    def prepare(self) -> Dict:
        """Prepare the corpus for BPE training.

        Returns:
            Preparation report dictionary.
        """
        self.logger.info("=" * 60)
        self.logger.info("Stage 3: BPE Corpus Preparation")
        self.logger.info("=" * 60)

        # Load and validate documents
        self.logger.info("Loading Stage 2 cleaned corpus...")
        documents = self.loader.load_corpus()
        validation_report = self.loader.validate_documents(documents)

        self.logger.info(
            f"Loaded {validation_report['documents_loaded']} documents, "
            f"{validation_report['documents_valid']} valid"
        )

        # Split documents
        self.logger.info(f"Splitting documents using {self.config.split_strategy} strategy...")
        splits = self.splitter.split(documents)

        # Check for leakage
        leakage_report = self.splitter.check_leakage(splits)
        if leakage_report["has_leakage"]:
            self.logger.error("Data leakage detected!")
            for leak in leakage_report["document_leakage"]:
                self.logger.error(f"  {leak}")
            for leak in leakage_report["company_leakage"]:
                self.logger.error(f"  {leak}")

        # Generate statistics
        self.logger.info("Generating corpus statistics...")
        corpus_stats = self.statistics.analyze_documents(documents)
        split_stats = self.statistics.analyze_split_statistics(splits)

        # Generate manifest
        self.logger.info("Generating split manifest...")
        company_assignments = getattr(self.splitter, "company_assignments", None)
        manifest = self.manifest_generator.generate(splits, company_assignments)

        # Write output files
        self.logger.info("Writing output files...")
        self._write_split_files(splits)
        self._write_metadata_files(splits)
        self._write_statistics(corpus_stats, split_stats)
        self._write_manifest(manifest)
        self._write_preparation_report(
            validation_report,
            leakage_report,
            corpus_stats,
            split_stats,
        )

        self.logger.info("=" * 60)
        self.logger.info("Stage 3 completed successfully")
        self.logger.info("=" * 60)

        return {
            "status": "success",
            "validation": validation_report,
            "leakage": leakage_report,
            "statistics": corpus_stats,
            "split_statistics": split_stats,
        }

    def _write_split_files(self, splits: Dict[str, List[Dict]]) -> None:
        """Write train/validation/test text files.

        Args:
            splits: Dictionary with train/validation/test document lists.
        """
        for split_name, documents in splits.items():
            output_path = self.output_dir / f"{split_name}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as f:
                for doc in documents:
                    text = doc["text"]

                    # Apply boundary mode
                    if self.config.boundary_mode == "special":
                        # Keep document boundaries as special tokens
                        f.write("<DOCUMENT_START>\n")
                        f.write(text)
                        f.write("\n<DOCUMENT_END>\n\n")
                    else:
                        # Remove boundaries, just use text
                        f.write(text)
                        f.write("\n\n")

            self.logger.info(f"Wrote {split_name}.txt ({len(documents)} documents)")

    def _write_metadata_files(self, splits: Dict[str, List[Dict]]) -> None:
        """Write metadata JSONL files for each split.

        Args:
            splits: Dictionary with train/validation/test document lists.
        """
        for split_name, documents in splits.items():
            output_path = self.output_dir / f"{split_name}_metadata.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as f:
                for doc in documents:
                    metadata = doc["metadata"].copy()
                    metadata["document_id"] = doc["document_id"]
                    metadata["length"] = len(doc["text"])
                    f.write(json.dumps(metadata) + "\n")

            self.logger.info(f"Wrote {split_name}_metadata.jsonl")

    def _write_statistics(
        self,
        corpus_stats: Dict,
        split_stats: Dict,
    ) -> None:
        """Write corpus statistics.

        Args:
            corpus_stats: Overall corpus statistics.
            split_stats: Per-split statistics.
        """
        output_path = self.output_dir / "corpus_statistics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "corpus_statistics": corpus_stats,
            "split_statistics": split_stats,
        }

        output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        self.logger.info("Wrote corpus_statistics.json")

    def _write_manifest(self, manifest: Dict) -> None:
        """Write split manifest.

        Args:
            manifest: Manifest dictionary.
        """
        output_path = self.output_dir / "split_manifest.json"
        self.manifest_generator.save_manifest(manifest, output_path)
        self.logger.info("Wrote split_manifest.json")

    def _write_preparation_report(
        self,
        validation_report: Dict,
        leakage_report: Dict,
        corpus_stats: Dict,
        split_stats: Dict,
    ) -> None:
        """Write preparation report.

        Args:
            validation_report: Document validation report.
            leakage_report: Leakage check report.
            corpus_stats: Corpus statistics.
            split_stats: Split statistics.
        """
        output_path = self.output_dir / "preparation_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "stage": 3,
            "status": "success",
            "configuration": {
                "split_strategy": self.config.split_strategy,
                "train_ratio": self.config.train_ratio,
                "validation_ratio": self.config.validation_ratio,
                "test_ratio": self.config.test_ratio,
                "seed": self.config.seed,
                "preserve_case": self.config.preserve_case,
                "preserve_punctuation": self.config.preserve_punctuation,
                "boundary_mode": self.config.boundary_mode,
                "min_document_length": self.config.min_document_length,
                "max_document_length": self.config.max_document_length,
            },
            "input": {
                "input_dir": str(self.input_dir),
                "documents_loaded": validation_report["documents_loaded"],
                "documents_valid": validation_report["documents_valid"],
                "documents_skipped": validation_report["documents_skipped"],
                "documents_failed": validation_report["documents_failed"],
            },
            "splits": {
                "train": split_stats.get("train", {}),
                "validation": split_stats.get("validation", {}),
                "test": split_stats.get("test", {}),
            },
            "leakage_check": leakage_report,
            "corpus_summary": {
                "total_documents": corpus_stats["document_statistics"]["number_of_documents"],
                "total_companies": corpus_stats["document_statistics"]["number_of_companies"],
                "total_characters": corpus_stats["character_statistics"]["total_characters"],
                "total_bytes": corpus_stats["byte_statistics"]["total_bytes"],
                "unique_characters": corpus_stats["character_statistics"]["unique_characters"],
            },
            "warnings": validation_report.get("warnings", []) + leakage_report.get("document_leakage", []),
            "errors": validation_report.get("errors", []),
        }

        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.logger.info("Wrote preparation_report.json")
