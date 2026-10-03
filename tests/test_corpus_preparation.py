"""Tests for Stage 3 corpus preparation."""

import json
import tempfile
from pathlib import Path

from sec_bpe.corpus.config import CorpusPreparationConfig
from sec_bpe.corpus.cleaned_corpus_loader import CleanedCorpusLoader
from sec_bpe.corpus.splitter import CorpusSplitter
from sec_bpe.corpus.statistics import CorpusStatistics
from sec_bpe.corpus.manifest import ManifestGenerator


class TestCorpusPreparationConfig:
    """Test cases for CorpusPreparationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CorpusPreparationConfig()
        assert config.split_strategy == "company"
        assert config.train_ratio == 0.80
        assert config.validation_ratio == 0.10
        assert config.test_ratio == 0.10
        assert config.seed == 42
        assert config.preserve_case is True
        assert config.preserve_punctuation is True
        assert config.boundary_mode == "special"

    def test_ratio_validation(self):
        """Test that invalid ratios raise ValueError."""
        try:
            CorpusPreparationConfig(train_ratio=0.5, validation_ratio=0.3, test_ratio=0.1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_valid_ratios(self):
        """Test that valid ratios are accepted."""
        config = CorpusPreparationConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)
        assert config.train_ratio == 0.7

    def test_invalid_split_strategy(self):
        """Test that invalid split strategy raises ValueError."""
        try:
            CorpusPreparationConfig(split_strategy="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_boundary_mode(self):
        """Test that invalid boundary mode raises ValueError."""
        try:
            CorpusPreparationConfig(boundary_mode="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestCleanedCorpusLoader:
    """Test cases for CleanedCorpusLoader."""

    def test_load_valid_corpus(self):
        """Test loading a valid corpus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test corpus
            corpus_path = tmpdir / "corpus.txt"
            corpus_path.write_text(
                "<DOCUMENT_START>\nTest document 1\n<DOCUMENT_END>\n\n"
                "<DOCUMENT_START>\nTest document 2\n<DOCUMENT_END>\n",
                encoding="utf-8",
            )

            # Create test metadata
            metadata_path = tmpdir / "metadata.jsonl"
            metadata_path.write_text(
                '{"document_id": "doc_000000", "status": "success"}\n'
                '{"document_id": "doc_000001", "status": "success"}\n',
                encoding="utf-8",
            )

            loader = CleanedCorpusLoader(tmpdir)
            documents = loader.load_corpus()

            assert len(documents) == 2
            assert documents[0]["document_id"] == "doc_000000"
            assert documents[0]["text"] == "Test document 1"

    def test_skip_failed_documents(self):
        """Test that failed documents are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            corpus_path = tmpdir / "corpus.txt"
            corpus_path.write_text(
                "<DOCUMENT_START>\nTest document\n<DOCUMENT_END>\n",
                encoding="utf-8",
            )

            metadata_path = tmpdir / "metadata.jsonl"
            metadata_path.write_text(
                '{"document_id": "doc_000000", "status": "failed"}\n',
                encoding="utf-8",
            )

            loader = CleanedCorpusLoader(tmpdir)
            documents = loader.load_corpus()

            assert len(documents) == 0

    def test_skip_short_documents(self):
        """Test that very short documents are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            corpus_path = tmpdir / "corpus.txt"
            corpus_path.write_text(
                "<DOCUMENT_START>\nShort\n<DOCUMENT_END>\n",
                encoding="utf-8",
            )

            metadata_path = tmpdir / "metadata.jsonl"
            metadata_path.write_text(
                '{"document_id": "doc_000000", "status": "success"}\n',
                encoding="utf-8",
            )

            loader = CleanedCorpusLoader(tmpdir)
            documents = loader.load_corpus()

            assert len(documents) == 0


class TestCorpusSplitter:
    """Test cases for CorpusSplitter."""

    def test_document_split(self):
        """Test document-level splitting."""
        config = CorpusPreparationConfig(split_strategy="document", seed=42)
        splitter = CorpusSplitter(config)

        documents = [
            {"document_id": f"doc_{i}", "text": f"Text {i}", "metadata": {}}
            for i in range(10)
        ]

        splits = splitter.split(documents)

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits
        assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == 10

    def test_company_split(self):
        """Test company-level splitting."""
        config = CorpusPreparationConfig(split_strategy="company", seed=42)
        splitter = CorpusSplitter(config)

        documents = []
        for i in range(9):
            ticker = f"TICK{i // 3}"  # 3 companies, 3 docs each
            documents.append({
                "document_id": f"doc_{i}",
                "text": f"Text {i}",
                "metadata": {"ticker": ticker},
            })

        splits = splitter.split(documents)

        # Check that all docs from a company are in the same split
        company_splits = {}
        for doc in documents:
            ticker = doc["metadata"]["ticker"]
            for split_name, split_docs in splits.items():
                if doc in split_docs:
                    company_splits[ticker] = split_name
                    break

        # Each company should be in exactly one split
        assert len(set(company_splits.values())) <= 3

    def test_leakage_check(self):
        """Test leakage detection."""
        config = CorpusPreparationConfig(split_strategy="document", seed=42)
        splitter = CorpusSplitter(config)

        documents = [
            {"document_id": f"doc_{i}", "text": f"Text {i}", "metadata": {}}
            for i in range(10)
        ]

        splits = splitter.split(documents)
        leakage_report = splitter.check_leakage(splits)

        assert leakage_report["has_leakage"] is False

    def test_deterministic_splitting(self):
        """Test that splitting is deterministic with same seed."""
        config = CorpusPreparationConfig(split_strategy="document", seed=42)

        documents = [
            {"document_id": f"doc_{i}", "text": f"Text {i}", "metadata": {}}
            for i in range(100)
        ]

        splitter1 = CorpusSplitter(config)
        splits1 = splitter1.split(documents)

        splitter2 = CorpusSplitter(config)
        splits2 = splitter2.split(documents)

        # Same document IDs in each split
        assert set(d["document_id"] for d in splits1["train"]) == set(d["document_id"] for d in splits2["train"])


class TestCorpusStatistics:
    """Test cases for CorpusStatistics."""

    def test_character_statistics(self):
        """Test character-level statistics."""
        stats = CorpusStatistics()

        documents = [
            {"document_id": "doc_1", "text": "Hello World", "metadata": {}},
            {"document_id": "doc_2", "text": "Test Text", "metadata": {}},
        ]

        report = stats.analyze_documents(documents)

        assert report["character_statistics"]["total_characters"] == 22
        assert report["character_statistics"]["unique_characters"] > 0

    def test_document_length_statistics(self):
        """Test document length statistics."""
        stats = CorpusStatistics()

        documents = [
            {"document_id": "doc_1", "text": "A" * 100, "metadata": {}},
            {"document_id": "doc_2", "text": "B" * 200, "metadata": {}},
            {"document_id": "doc_3", "text": "C" * 300, "metadata": {}},
        ]

        report = stats.analyze_documents(documents)

        assert report["document_length_statistics"]["minimum_document_length"] == 100
        assert report["document_length_statistics"]["maximum_document_length"] == 300
        assert report["document_length_statistics"]["mean_document_length"] == 200

    def test_financial_pattern_statistics(self):
        """Test financial pattern counting."""
        stats = CorpusStatistics()

        documents = [
            {"document_id": "doc_1", "text": "Revenue increased by 12%", "metadata": {}},
            {"document_id": "doc_2", "text": "$1.2 billion in assets", "metadata": {}},
        ]

        report = stats.analyze_documents(documents)

        assert report["financial_pattern_statistics"]["%"] > 0
        assert report["financial_pattern_statistics"]["$"] > 0

    def test_split_statistics(self):
        """Test per-split statistics."""
        stats = CorpusStatistics()

        splits = {
            "train": [
                {"document_id": "doc_1", "text": "A" * 1000, "metadata": {"ticker": "TICK1"}},
            ],
            "validation": [
                {"document_id": "doc_2", "text": "B" * 500, "metadata": {"ticker": "TICK2"}},
            ],
        }

        split_stats = stats.analyze_split_statistics(splits)

        assert split_stats["train"]["documents"] == 1
        assert split_stats["train"]["characters"] == 1000
        assert split_stats["validation"]["documents"] == 1
        assert split_stats["validation"]["characters"] == 500


class TestManifestGenerator:
    """Test cases for ManifestGenerator."""

    def test_generate_manifest(self):
        """Test manifest generation."""
        config = CorpusPreparationConfig(split_strategy="document", seed=42)
        generator = ManifestGenerator(config)

        splits = {
            "train": [
                {"document_id": "doc_1", "text": "Text 1", "metadata": {}},
                {"document_id": "doc_2", "text": "Text 2", "metadata": {}},
            ],
            "validation": [
                {"document_id": "doc_3", "text": "Text 3", "metadata": {}},
            ],
            "test": [
                {"document_id": "doc_4", "text": "Text 4", "metadata": {}},
            ],
        }

        manifest = generator.generate(splits)

        assert manifest["strategy"] == "document"
        assert manifest["seed"] == 42
        assert len(manifest["train"]) == 2
        assert len(manifest["validation"]) == 1
        assert len(manifest["test"]) == 1
        assert "doc_1" in manifest["train"]

    def test_save_and_load_manifest(self):
        """Test saving and loading manifest."""
        config = CorpusPreparationConfig(split_strategy="document", seed=42)
        generator = ManifestGenerator(config)

        splits = {
            "train": [{"document_id": "doc_1", "text": "Text", "metadata": {}}],
            "validation": [],
            "test": [],
        }

        manifest = generator.generate(splits)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = tmpdir / "manifest.json"

            generator.save_manifest(manifest, manifest_path)
            loaded_manifest = generator.load_manifest(manifest_path)

            assert loaded_manifest == manifest

    def test_validate_manifest(self):
        """Test manifest validation."""
        config = CorpusPreparationConfig(split_strategy="document", seed=42)
        generator = ManifestGenerator(config)

        documents = [
            {"document_id": "doc_1", "text": "Text 1", "metadata": {}},
            {"document_id": "doc_2", "text": "Text 2", "metadata": {}},
        ]

        splits = {
            "train": [documents[0]],
            "validation": [documents[1]],
            "test": [],
        }

        manifest = generator.generate(splits)
        validation_report = generator.validate_manifest(manifest, documents)

        assert validation_report["is_valid"] is True

    def test_detect_manifest_leakage(self):
        """Test that manifest validation detects leakage."""
        config = CorpusPreparationConfig(split_strategy="document", seed=42)
        generator = ManifestGenerator(config)

        documents = [
            {"document_id": "doc_1", "text": "Text 1", "metadata": {}},
        ]

        # Create manifest with leakage (same doc in train and validation)
        manifest = {
            "strategy": "document",
            "seed": 42,
            "train": ["doc_1"],
            "validation": ["doc_1"],
            "test": [],
        }

        validation_report = generator.validate_manifest(manifest, documents)

        assert validation_report["is_valid"] is False
        assert len(validation_report["errors"]) > 0
