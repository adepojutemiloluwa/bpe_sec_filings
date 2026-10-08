"""Tests for corpus loading and building."""

import tempfile
from pathlib import Path

import pytest

from sec_bpe.corpus.builder import CorpusBuilder
from sec_bpe.corpus.loader import CorpusLoader


class TestCorpusLoader:
    """Test cases for CorpusLoader."""

    def test_finds_supported_files(self):
        """Test that only supported file extensions are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test files
            (tmpdir / "file.txt").write_text("text content")
            (tmpdir / "file.html").write_text("<html>content</html>")
            (tmpdir / "file.htm").write_text("<html>content</html>")
            (tmpdir / "file.pdf").write_text("pdf content")
            (tmpdir / "file.csv").write_text("csv content")
            
            loader = CorpusLoader(tmpdir)
            documents = loader.find_documents()
            
            assert len(documents) == 3
            extensions = {doc.suffix for doc in documents}
            assert extensions == {".txt", ".html", ".htm"}

    def test_case_insensitive_extensions(self):
        """Test that file extensions are recognized case-insensitively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create files with mixed case extensions
            (tmpdir / "FILE.HTML").write_text("html content")
            (tmpdir / "FILE.HtM").write_text("htm content")
            (tmpdir / "FILE.TXT").write_text("txt content")
            
            loader = CorpusLoader(tmpdir)
            documents = loader.find_documents()
            
            assert len(documents) == 3

    def test_recursive_discovery(self):
        """Test that files are discovered recursively in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create nested directory structure
            (tmpdir / "company1").mkdir()
            (tmpdir / "company2").mkdir()
            (tmpdir / "company2" / "subdir").mkdir()
            
            # Create files in different directories
            (tmpdir / "company1" / "filing1.txt").write_text("filing 1")
            (tmpdir / "company2" / "filing2.html").write_text("filing 2")
            (tmpdir / "company2" / "subdir" / "filing3.htm").write_text("filing 3")
            
            loader = CorpusLoader(tmpdir)
            documents = loader.find_documents()
            
            assert len(documents) == 3

    def test_document_loading(self):
        """Test that documents are loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            test_file = tmpdir / "test.txt"
            test_content = "This is test content\nwith multiple lines"
            test_file.write_text(test_content)
            
            loader = CorpusLoader(tmpdir)
            loaded_content = loader.load_document(test_file)
            
            assert loaded_content == test_content

    def test_empty_directory(self):
        """Test that an informative error is raised for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            loader = CorpusLoader(tmpdir)
            
            with pytest.raises(ValueError, match="No supported documents found"):
                loader.load_all()

    def test_nonexistent_directory(self):
        """Test that FileNotFoundError is raised for nonexistent directory."""
        loader = CorpusLoader("/nonexistent/path")
        
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            loader.find_documents()

    def test_deterministic_ordering(self):
        """Test that documents are returned in deterministic order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create files in non-alphabetical order
            (tmpdir / "z.txt").write_text("z")
            (tmpdir / "a.txt").write_text("a")
            (tmpdir / "m.txt").write_text("m")
            
            loader = CorpusLoader(tmpdir)
            documents = loader.find_documents()
            
            # Should be sorted
            assert documents[0].name == "a.txt"
            assert documents[1].name == "m.txt"
            assert documents[2].name == "z.txt"


class TestCorpusBuilder:
    """Test cases for CorpusBuilder."""

    def test_corpus_building(self):
        """Test that multiple documents are combined into a corpus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create input directory with test files
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            
            (input_dir / "doc1.txt").write_text("Document 1 content")
            (input_dir / "doc2.html").write_text("<html>Document 2</html>")
            (input_dir / "doc3.htm").write_text("<html>Document 3</html>")
            
            # Create output path
            output_path = tmpdir / "output" / "corpus.txt"
            
            builder = CorpusBuilder(input_dir, output_path)
            count = builder.build()
            
            assert count == 3
            assert output_path.exists()
            
            # Verify content
            corpus_content = output_path.read_text()
            assert "<DOCUMENT_START>" in corpus_content
            assert "<DOCUMENT_END>" in corpus_content
            assert "Document 1 content" in corpus_content
            assert "<html>Document 2</html>" in corpus_content
            assert "<html>Document 3</html>" in corpus_content

    def test_document_separation(self):
        """Test that documents are properly separated with markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            
            (input_dir / "doc1.txt").write_text("DOC1")
            (input_dir / "doc2.txt").write_text("DOC2")
            
            output_path = tmpdir / "corpus.txt"
            
            builder = CorpusBuilder(input_dir, output_path)
            builder.build()
            
            corpus_content = output_path.read_text()
            
            # Documents should be separated with markers
            expected = "<DOCUMENT_START>\nDOC1\n<DOCUMENT_END>\n\n<DOCUMENT_START>\nDOC2\n<DOCUMENT_END>"
            assert corpus_content == expected

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            (input_dir / "doc.txt").write_text("content")
            
            # Output path with non-existent parent directory
            output_path = tmpdir / "nonexistent" / "nested" / "corpus.txt"
            
            builder = CorpusBuilder(input_dir, output_path)
            builder.build()
            
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_single_document(self):
        """Test building corpus with a single document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            (input_dir / "doc.txt").write_text("Single document")
            
            output_path = tmpdir / "corpus.txt"
            
            builder = CorpusBuilder(input_dir, output_path)
            count = builder.build()
            
            assert count == 1
            expected = "<DOCUMENT_START>\nSingle document\n<DOCUMENT_END>"
            assert output_path.read_text() == expected

    def test_no_supported_files_error(self):
        """Test that ValueError is raised when no supported files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            (input_dir / "doc.pdf").write_text("pdf content")
            
            output_path = tmpdir / "corpus.txt"
            
            builder = CorpusBuilder(input_dir, output_path)
            
            with pytest.raises(ValueError, match="No supported documents found"):
                builder.build()

    def test_raw_content_preserved(self):
        """Test that raw content is preserved without preprocessing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            
            # Create HTML file with tags, tables, etc.
            html_content = """
            <html>
                <body>
                    <table>
                        <tr><td>Data</td></tr>
                    </table>
                    <p>Text with <b>formatting</b></p>
                    Numbers: 123.45
                    Symbols: $, %, &
                </body>
            </html>
            """
            (input_dir / "doc.html").write_text(html_content)
            
            output_path = tmpdir / "corpus.txt"
            
            builder = CorpusBuilder(input_dir, output_path)
            builder.build()
            
            corpus_content = output_path.read_text()
            
            # Verify raw content is preserved with markers
            assert "<DOCUMENT_START>" in corpus_content
            assert "<DOCUMENT_END>" in corpus_content
            assert "<html>" in corpus_content
            assert "<table>" in corpus_content
            assert "123.45" in corpus_content
            assert "$, %, &" in corpus_content
