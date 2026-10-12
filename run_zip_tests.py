"""Simple test runner for ZIP extraction tests (no pytest required)."""

import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sec_bpe.corpus.ingestion_config import IngestionConfig
from sec_bpe.corpus.zip_extractor import ZipExtractor
from sec_bpe.corpus.loader import CorpusLoader
from sec_bpe.corpus.builder import CorpusBuilder


def run_test(test_name, test_func):
    """Run a single test function."""
    try:
        test_func()
        print(f"[PASS] {test_name}")
        return True
    except AssertionError as e:
        print(f"[FAIL] {test_name}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {test_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_extraction_enabled():
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


def test_extraction_disabled():
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


def test_multiple_files_in_zip():
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


def test_corrupt_zip_archive():
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


def test_empty_zip_archive():
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


def test_duplicate_filenames_different_archives():
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


def test_path_traversal_attack():
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


def test_unsupported_files_in_zip():
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


def test_rerun_without_duplicate_extraction():
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


def test_file_size_limit():
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


def test_total_size_limit():
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


def test_loader_with_zip_enabled():
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


def test_loader_with_zip_disabled():
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
        assert len(documents) == 1, f"Expected 1 document, found {len(documents)}"
        assert documents[0].name == "doc2.txt", f"Expected doc2.txt, found {documents[0].name}"

        # Verify extraction report
        assert loader.extraction_report is not None, "Extraction report is None"
        assert loader.extraction_report["enabled"] is False, f"Expected enabled=False, got {loader.extraction_report['enabled']}"


def test_regression_non_zip_ingestion():
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


def test_builder_with_zip_extraction():
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


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running ZIP Extraction Tests")
    print("=" * 60)

    tests = [
        ("ZIP extraction enabled", test_extraction_enabled),
        ("ZIP extraction disabled", test_extraction_disabled),
        ("Multiple files in ZIP", test_multiple_files_in_zip),
        ("Corrupt ZIP archive", test_corrupt_zip_archive),
        ("Empty ZIP archive", test_empty_zip_archive),
        ("Duplicate filenames in different archives", test_duplicate_filenames_different_archives),
        ("Path traversal attack prevention", test_path_traversal_attack),
        ("Unsupported files in ZIP", test_unsupported_files_in_zip),
        ("Rerun without duplicate extraction", test_rerun_without_duplicate_extraction),
        ("File size limit", test_file_size_limit),
        ("Total size limit (ZIP bomb)", test_total_size_limit),
        ("Loader with ZIP enabled", test_loader_with_zip_enabled),
        ("Loader with ZIP disabled", test_loader_with_zip_disabled),
        ("Regression: non-ZIP ingestion", test_regression_non_zip_ingestion),
        ("Builder with ZIP extraction", test_builder_with_zip_extraction),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
