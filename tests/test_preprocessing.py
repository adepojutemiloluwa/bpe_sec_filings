"""Tests for Stage 2 preprocessing components."""

import tempfile
from pathlib import Path

import pytest

from sec_bpe.preprocessing.encoding import EncodingNormalizer
from sec_bpe.preprocessing.html_cleaner import HTMLCleaner
from sec_bpe.preprocessing.xbrl_cleaner import XBRLCleaner
from sec_bpe.preprocessing.table_parser import TableParser
from sec_bpe.preprocessing.structure import StructurePreserver
from sec_bpe.preprocessing.whitespace import WhitespaceNormalizer
from sec_bpe.preprocessing.deduplication import DuplicateDetector
from sec_bpe.preprocessing.document_cleaner import DocumentCleaner


class TestEncodingNormalizer:
    """Test cases for EncodingNormalizer."""

    def test_html_entity_decoding(self):
        """Test HTML entity decoding."""
        normalizer = EncodingNormalizer()
        text = "Revenue increased by &lt;12%&gt;"
        result = normalizer.normalize(text)
        assert "<12%>" in result

    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        normalizer = EncodingNormalizer()
        text = "café"
        result = normalizer.normalize(text)
        assert len(result) > 0

    def test_special_char_normalization(self):
        """Test special character normalization."""
        normalizer = EncodingNormalizer()
        text = "smart quotes "test" and dashes —"
        result = normalizer.normalize(text)
        assert '"' in result
        assert '--' in result

    def test_non_breaking_space_removal(self):
        """Test non-breaking space removal."""
        normalizer = EncodingNormalizer()
        text = "text\u00A0with\u00A0spaces"
        result = normalizer.normalize(text)
        assert '\u00A0' not in result


class TestHTMLCleaner:
    """Test cases for HTMLCleaner."""

    def test_simple_tag_removal(self):
        """Test simple HTML tag removal."""
        cleaner = HTMLCleaner()
        text = "<p>Revenue increased by <b>12%</b>.</p>"
        result = cleaner.clean(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "12%" in result

    def test_script_removal(self):
        """Test script tag removal with content."""
        cleaner = HTMLCleaner()
        text = "<p>Text</p><script>alert('test');</script><p>More text</p>"
        result = cleaner.clean(text)
        assert "<script>" not in result
        assert "alert" not in result
        assert "Text" in result
        assert "More text" in result

    def test_comment_removal(self):
        """Test HTML comment removal."""
        cleaner = HTMLCleaner()
        text = "<p>Text</p><!-- comment --><p>More text</p>"
        result = cleaner.clean(text)
        assert "<!--" not in result
        assert "comment" not in result
        assert "Text" in result

    def test_self_closing_tag_removal(self):
        """Test self-closing tag removal."""
        cleaner = HTMLCleaner()
        text = "<p>Text<br/>More text</p>"
        result = cleaner.clean(text)
        assert "<br/>" not in result

    def test_stats_tracking(self):
        """Test statistics tracking."""
        cleaner = HTMLCleaner()
        text = "<p>Text</p><b>Bold</b>"
        cleaner.clean(text)
        stats = cleaner.get_stats()
        assert stats["elements_removed"] > 0


class TestXBRLCleaner:
    """Test cases for XBRLCleaner."""

    def test_context_removal(self):
        """Test XBRL context element removal."""
        cleaner = XBRLCleaner()
        text = "<xbrli:context id='ctx1'>...</xbrli:context>Text"
        result = cleaner.clean(text)
        assert "<xbrli:context" not in result
        assert "Text" in result

    def test_unit_removal(self):
        """Test XBRL unit element removal."""
        cleaner = XBRLCleaner()
        text = "<xbrli:unit id='USD'>...</xbrli:unit>Text"
        result = cleaner.clean(text)
        assert "<xbrli:unit" not in result

    def test_namespace_tag_removal(self):
        """Test XBRL namespace tag removal."""
        cleaner = XBRLCleaner()
        text = "<us-gaap:Revenue>100</us-gaap:Revenue>Text"
        result = cleaner.clean(text)
        assert "<us-gaap:Revenue" not in result
        assert "100" in result

    def test_stats_tracking(self):
        """Test statistics tracking."""
        cleaner = XBRLCleaner()
        text = "<xbrli:context>...</xbrli:context>"
        cleaner.clean(text)
        stats = cleaner.get_stats()
        assert stats["xbrl_elements_removed"] > 0


class TestTableParser:
    """Test cases for TableParser."""

    def test_simple_table_extraction(self):
        """Test simple table extraction."""
        parser = TableParser()
        text = "<table><tr><td>Revenue</td><td>$100M</td></tr></table>"
        result = parser.extract_tables(text)
        assert "<table>" not in result
        assert "Revenue" in result
        assert "$100M" in result

    def test_multiple_rows(self):
        """Test table with multiple rows."""
        parser = TableParser()
        text = "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>"
        result = parser.extract_tables(text)
        assert "A" in result
        assert "B" in result

    def test_table_count(self):
        """Test table count tracking."""
        parser = TableParser()
        text = "<table><tr><td>A</td></tr></table><table><tr><td>B</td></tr></table>"
        parser.extract_tables(text)
        stats = parser.get_stats()
        assert stats["table_count"] == 2


class TestStructurePreserver:
    """Test cases for StructurePreserver."""

    def test_part_heading_preservation(self):
        """Test PART heading preservation."""
        preserver = StructurePreserver()
        text = "Text\nPART I\nMore text"
        result = preserver.preserve(text)
        assert "PART I" in result

    def test_item_heading_preservation(self):
        """Test ITEM heading preservation."""
        preserver = StructurePreserver()
        text = "Text\nITEM 1. BUSINESS\nMore text"
        result = preserver.preserve(text)
        assert "ITEM 1. BUSINESS" in result


class TestWhitespaceNormalizer:
    """Test cases for WhitespaceNormalizer."""

    def test_excessive_blank_lines(self):
        """Test removal of excessive blank lines."""
        normalizer = WhitespaceNormalizer()
        text = "Line 1\n\n\n\nLine 2"
        result = normalizer.normalize(text)
        assert "\n\n\n" not in result

    def test_trailing_whitespace(self):
        """Test trailing whitespace removal."""
        normalizer = WhitespaceNormalizer()
        text = "Line 1  \nLine 2\t"
        result = normalizer.normalize(text)
        assert "Line 1\nLine 2" in result

    def test_multiple_spaces(self):
        """Test multiple space normalization."""
        normalizer = WhitespaceNormalizer()
        text = "Word 1   Word 2"
        result = normalizer.normalize(text)
        assert "Word 1 Word 2" in result

    def test_page_artifact_removal(self):
        """Test page artifact removal."""
        normalizer = WhitespaceNormalizer()
        text = "Text\nPage 1 of 10\nMore text"
        result = normalizer.remove_extraction_artifacts(text)
        assert "Page 1 of 10" not in result


class TestDuplicateDetector:
    """Test cases for DuplicateDetector."""

    def test_exact_duplicate_detection(self):
        """Test exact duplicate detection."""
        detector = DuplicateDetector()
        text = "This is a test document."
        
        is_dup1, orig1 = detector.check_duplicate("doc1", text)
        is_dup2, orig2 = detector.check_duplicate("doc2", text)
        
        assert not is_dup1
        assert is_dup2
        assert orig2 == "doc1"

    def test_unique_documents(self):
        """Test unique documents are not flagged as duplicates."""
        detector = DuplicateDetector()
        
        is_dup1, _ = detector.check_duplicate("doc1", "Text A")
        is_dup2, _ = detector.check_duplicate("doc2", "Text B")
        
        assert not is_dup1
        assert not is_dup2

    def test_case_insensitive_hashing(self):
        """Test that hashing is case-insensitive."""
        detector = DuplicateDetector()
        
        is_dup1, _ = detector.check_duplicate("doc1", "Test Document")
        is_dup2, _ = detector.check_duplicate("doc2", "test document")
        
        assert is_dup2

    def test_stats_tracking(self):
        """Test statistics tracking."""
        detector = DuplicateDetector()
        detector.check_duplicate("doc1", "Text A")
        detector.check_duplicate("doc2", "Text A")
        
        stats = detector.get_stats()
        assert stats["unique_documents"] == 1
        assert stats["duplicate_documents"] == 1


class TestDocumentCleaner:
    """Test cases for DocumentCleaner."""

    def test_full_pipeline(self):
        """Test full cleaning pipeline."""
        cleaner = DocumentCleaner()
        text = "<p>Revenue increased by <b>12%</b>.</p>"
        
        cleaned, metadata = cleaner.clean_document("test_doc", text)
        
        assert metadata["status"] == "success"
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "12%" in cleaned

    def test_metadata_generation(self):
        """Test metadata generation."""
        cleaner = DocumentCleaner()
        text = "Test document"
        
        cleaned, metadata = cleaner.clean_document("test_doc", text)
        
        assert "document_id" in metadata
        assert "original_length" in metadata
        assert "cleaned_length" in metadata
        assert "steps_completed" in metadata

    def test_error_handling(self):
        """Test error handling for malformed input."""
        cleaner = DocumentCleaner()
        
        # This should not crash
        cleaned, metadata = cleaner.clean_document("test_doc", "")
        assert metadata["status"] == "success"

    def test_duplicate_check(self):
        """Test duplicate checking through cleaner."""
        cleaner = DocumentCleaner()
        text = "Test document"
        
        is_dup1, _ = cleaner.check_duplicate("doc1", text)
        is_dup2, _ = cleaner.check_duplicate("doc2", text)
        
        assert not is_dup1
        assert is_dup2
