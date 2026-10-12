"""Tests for ZIP archive extraction."""

import tempfile
import zipfile
from pathlib import Path

import pytest

from sec_bpe.corpus.ingestion_config import IngestionConfig
from sec_bpe.corpus.zip_extractor import ZipExtractor
from sec_bpe.corpus.loader import CorpusLoader
from sec_bpe.corpus.builder import CorpusBuilder


class TestZipExtractor:
    """Test cases for ZipExtractor."""

    def test_extraction_enabled(self):
        """Test that extraction works when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a ZIP file with test documents
            zip_path = tmpdir / "test.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("file1.txt", "Content 1")
                zf.writestr("file2.html", "<html>Content 2</html>")

            # Configure extraction
            config = IngestionConfig(extract_zip=True)
            extractor = ZipExtractor(config)

            # Extract
            report = extractor.extract_archives(tmpdir)

            # Verify
            assert report["enabled"] is True
            assert report["archives_discovered"] == 1
            assert report["archives_extracted"] == 1
            assert report["files_extracted"] == 2

            # Verify files exist
            extract_dir = tmpdir / "_extracted" / "test"
            assert (extract_dir / "file1.txt").exists()
            assert (extract_dir / "file2.html").exists()

    def test_extraction_disabled(self):
        """Test that extraction is skipped when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a ZIP file
            zip_path = tmpdir / "test.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("file1.txt", "Content 1")

            # Configure extraction disabled
            config = IngestionConfig(extract_zip=False)
            extractor = ZipExtractor(config)

            # Extract
            report = extractor.extract_archives(tmpdir)

            # Verify
            assert report["enabled"] is False
            assert report["archives_discovered"] == 0
            assert report["archives_extracted"] == 0
            assert report["files_extracted"] == 0

            # Verify no extraction directory
            assert not (tmpdir / "_extracted").exists()

    def test_multiple_files_in_zip(self):
        """Test extraction of ZIP with multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create ZIP with multiple files
            zip_path = tmpdir / "multi.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for i in range(10):
                    zf.writestr(f"file{i}.txt", f"Content {i}")

            config = IngestionConfig(extract_zip=True)
            extractor = ZipExtractor(config)
            report = extractor.extract_archives(tmpdir)

            assert report["archives_extracted"] == 1
            assert report["files_extracted"] == 10

    def test_corrupt_zip_archive(self):
        """Test handling of corrupt ZIP archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a file that looks like ZIP but is corrupt
            zip_path = tmpdir / "corrupt.zip"
            zip_path.write_bytes(b"not a valid zip file")

            config = IngestionConfig(extract_zip=True)
            extractor = ZipExtractor(config)
            report = extractor.extract_archives(tmpdir)

            assert report["archives_discovered"] == 1
            assert report["archives_failed"] == 1
            assert report["archives_extracted"] == 0
            assert len(report["errors"]) == 1

    def test_empty_zip_archive(self):
        """Test handling of empty ZIP archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create empty ZIP
            zip_path = tmpdir / "empty.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                pass  # Empty archive

            config = IngestionConfig(extract_zip=True)
            extractor = ZipExtractor(config)
            report = extractor.extract_archives(tmpdir)

            assert report["archives_extracted"] == 1
            assert report["files_extracted"] == 0

    def test_duplicate_filenames_different_archives(self):
        """Test handling of duplicate filenames across archives."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create two ZIPs with same filename
            zip1 = tmpdir / "archive1.zip"
            zip2 = tmpdir / "archive2.zip"

            with zipfile.ZipFile(zip1, 'w') as zf:
                zf.writestr("file.txt", "Content from archive 1")

            with zipfile.ZipFile(zip2, 'w') as zf:
                zf.writestr("file.txt", "Content from archive 2")

            config = IngestionConfig(extract_zip=True)
            extractor = ZipExtractor(config)
            report = extractor.extract_archives(tmpdir)

            assert report["archives_extracted"] == 2
            assert report["files_extracted"] == 2

            # Verify both files exist with different names
            extract_dir = tmpdir / "_extracted"
            assert (extract_dir / "archive1" / "file.txt").exists()
            assert (extract_dir / "archive2" / "file.txt").exists()

    def test_path_traversal_attack(self):
        """Test that path traversal attempts are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create ZIP with malicious path
            zip_path = tmpdir / "malicious.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("../etc/passwd", "malicious content")
                zf.writestr("C:\\Windows\\System32\\config", "malicious content")

            config = IngestionConfig(extract_zip=True)
            extractor = ZipExtractor(config)
            report = extractor.extract_archives(tmpdir)

            assert report["archives_extracted"] == 1
            assert report["files_extracted"] == 0

            # Verify no files escaped extraction directory
            assert not (tmpdir / "etc").exists()
            assert not (tmpdir / "Windows").exists()

    def test_unsupported_files_in_zip(self):
        """Test that unsupported files are extracted but not processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create ZIP with mixed file types
            zip_path = tmpdir / "mixed.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("file.txt", "text content")
                zf.writestr("file.pdf", "pdf content")
                zf.writestr("file.csv", "csv content")

            config = IngestionConfig(extract_zip=True)
            extractor = ZipExtractor(config)
            report = extractor.extract_archives(tmpdir)

            # All files should be extracted
            assert report["files_extracted"] == 3

            # But loader should only pick up supported files
            loader = CorpusLoader(tmpdir, config)
            documents = loader.find_documents()

            # Only .txt should be found
            assert len(documents) == 1
            assert documents[0].suffix == ".txt"

    def test_rerun_without_duplicate_extraction(self):
        """Test that rerunning doesn't re-extract unchanged archives."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create ZIP
            zip_path = tmpdir / "test.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("file.txt", "Content")

            config = IngestionConfig(extract_zip=True)
            extractor = ZipExtractor(config)

            # First extraction
            report1 = extractor.extract_archives(tmpdir)
            assert report1["archives_extracted"] == 1
            assert report1["archives_skipped"] == 0

            # Second extraction (should skip)
            report2 = extractor.extract_archives(tmpdir)
            assert report2["archives_extracted"] == 0
            assert report2["archives_skipped"] == 1

    def test_file_size_limit(self):
        """Test that oversized files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create ZIP with large file
            zip_path = tmpdir / "large.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                # 1MB file
                large_content = b"x" * (1024 * 1024)
                zf.writestr("large.txt", large_content)
                zf.writestr("small.txt", b"small")

            # Set low limit
            config = IngestionConfig(extract_zip=True, max_extracted_file_size=1000)
            extractor = ZipExtractor(config)
            report = extractor.extract_archives(tmpdir)

            assert report["archives_extracted"] == 1
            assert report["files_extracted"] == 1  # Only small file

    def test_total_size_limit(self):
        """Test that ZIP bombs are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create ZIP with many files
            zip_path = tmpdir / "bomb.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for i in range(100):
                    zf.writestr(f"file{i}.txt", b"x" * 10000)

            # Set low total limit
            config = IngestionConfig(extract_zip=True, max_total_extracted_size=50000)
            extractor = ZipExtractor(config)

            # Should raise error for ZIP bomb
            report = extractor.extract_archives(tmpdir)
            assert report["archives_failed"] == 1
            assert "ZIP bomb" in str(report["errors"])


class TestCorpusLoaderIntegration:
    """Test integration of ZIP extraction with CorpusLoader."""

    def test_loader_with_zip_enabled(self):
        """Test that loader extracts ZIP when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create ZIP with supported files
            zip_path = tmpdir / "docs.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("doc1.txt", "Document 1")
                zf.writestr("doc2.html", "<html>Document 2</html>")

            # Create regular file
            (tmpdir / "doc3.txt").write_text("Document 3")

            config = IngestionConfig(extract_zip=True)
            loader = CorpusLoader(tmpdir, config)
            documents = loader.find_documents()

            # Should find 3 documents (2 from ZIP, 1 regular)
            assert len(documents) == 3

            # Verify extraction report
            assert loader.extraction_report is not None
            assert loader.extraction_report["archives_extracted"] == 1

    def test_loader_with_zip_disabled(self):
        """Test that loader skips ZIP when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create ZIP
            zip_path = tmpdir / "docs.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("doc1.txt", "Document 1")

            # Create regular file
            (tmpdir / "doc2.txt").write_text("Document 2")

            config = IngestionConfig(extract_zip=False)
            loader = CorpusLoader(tmpdir, config)
            documents = loader.find_documents()

            # Should only find regular file
            assert len(documents) == 1
            assert documents[0].name == "doc2.txt"

            # Verify extraction report
            assert loader.extraction_report is not None
            assert loader.extraction_report["enabled"] is False

    def test_regression_non_zip_ingestion(self):
        """Test that non-ZIP ingestion still works as before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create regular files
            (tmpdir / "doc1.txt").write_text("Document 1")
            (tmpdir / "doc2.html").write_text("<html>Document 2</html>")
            (tmpdir / "subdir").mkdir()
            (tmpdir / "subdir" / "doc3.txt").write_text("Document 3")

            config = IngestionConfig(extract_zip=False)
            loader = CorpusLoader(tmpdir, config)
            documents = loader.find_documents()

            # Should find all 3 files
            assert len(documents) == 3
            assert all(doc.suffix in {".txt", ".html"} for doc in documents)


class TestCorpusBuilderIntegration:
    """Test integration of ZIP extraction with CorpusBuilder."""

    def test_builder_with_zip_extraction(self):
        """Test that builder works with ZIP extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create ZIP
            zip_path = tmpdir / "docs.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("doc1.txt", "Document 1")
                zf.writestr("doc2.txt", "Document 2")

            # Build corpus
            output_path = tmpdir / "corpus.txt"
            config = IngestionConfig(extract_zip=True)
            builder = CorpusBuilder(tmpdir, output_path, config)
            count = builder.build()

            # Should process 2 documents
            assert count == 2

            # Verify corpus exists
            assert output_path.exists()
            content = output_path.read_text()
            assert "<DOCUMENT_START>" in content
            assert "<DOCUMENT_END>" in content
