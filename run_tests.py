"""Simple test runner for preprocessing tests (no pytest required)."""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sec_bpe.preprocessing.encoding import EncodingNormalizer
from sec_bpe.preprocessing.html_cleaner import HTMLCleaner
from sec_bpe.preprocessing.xbrl_cleaner import XBRLCleaner
from sec_bpe.preprocessing.table_parser import TableParser
from sec_bpe.preprocessing.structure import StructurePreserver
from sec_bpe.preprocessing.whitespace import WhitespaceNormalizer
from sec_bpe.preprocessing.deduplication import DuplicateDetector
from sec_bpe.preprocessing.document_cleaner import DocumentCleaner


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


def test_encoding_html_entity_decoding():
    normalizer = EncodingNormalizer()
    text = "Revenue increased by &lt;12%&gt;"
    result = normalizer.normalize(text)
    assert "<12%>" in result


def test_encoding_unicode_normalization():
    normalizer = EncodingNormalizer()
    text = "café"
    result = normalizer.normalize(text)
    assert len(result) > 0


def test_encoding_special_char_normalization():
    normalizer = EncodingNormalizer()
    text = 'smart quotes "test" and dashes —'
    result = normalizer.normalize(text)
    assert '"' in result
    assert '--' in result


def test_html_simple_tag_removal():
    cleaner = HTMLCleaner()
    text = "<p>Revenue increased by <b>12%</b>.</p>"
    result = cleaner.clean(text)
    assert "<p>" not in result
    assert "<b>" not in result
    assert "12%" in result


def test_html_script_removal():
    cleaner = HTMLCleaner()
    text = "<p>Text</p><script>alert('test');</script><p>More text</p>"
    result = cleaner.clean(text)
    assert "<script>" not in result
    assert "alert" not in result
    assert "Text" in result
    assert "More text" in result


def test_html_comment_removal():
    cleaner = HTMLCleaner()
    text = "<p>Text</p><!-- comment --><p>More text</p>"
    result = cleaner.clean(text)
    assert "<!--" not in result
    assert "comment" not in result
    assert "Text" in result


def test_xbrl_context_removal():
    cleaner = XBRLCleaner()
    text = "<xbrli:context id='ctx1'>...</xbrli:context>Text"
    result = cleaner.clean(text)
    assert "<xbrli:context" not in result
    assert "Text" in result


def test_xbrl_namespace_tag_removal():
    cleaner = XBRLCleaner()
    text = "<us-gaap:Revenue>100</us-gaap:Revenue>Text"
    result = cleaner.clean(text)
    assert "<us-gaap:Revenue" not in result
    assert "100" in result


def test_table_simple_extraction():
    parser = TableParser()
    text = "<table><tr><td>Revenue</td><td>$100M</td></tr></table>"
    result = parser.extract_tables(text)
    assert "<table>" not in result
    assert "Revenue" in result
    assert "$100M" in result


def test_table_multiple_rows():
    parser = TableParser()
    text = "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>"
    result = parser.extract_tables(text)
    assert "A" in result
    assert "B" in result


def test_structure_part_heading():
    preserver = StructurePreserver()
    text = "Text\nPART I\nMore text"
    result = preserver.preserve(text)
    assert "PART I" in result


def test_structure_item_heading():
    preserver = StructurePreserver()
    text = "Text\nITEM 1. BUSINESS\nMore text"
    result = preserver.preserve(text)
    assert "ITEM 1. BUSINESS" in result


def test_whitespace_excessive_blank_lines():
    normalizer = WhitespaceNormalizer()
    text = "Line 1\n\n\n\nLine 2"
    result = normalizer.normalize(text)
    assert "\n\n\n" not in result


def test_whitespace_trailing():
    normalizer = WhitespaceNormalizer()
    text = "Line 1  \nLine 2\t"
    result = normalizer.normalize(text)
    assert "Line 1\nLine 2" in result


def test_whitespace_multiple_spaces():
    normalizer = WhitespaceNormalizer()
    text = "Word 1   Word 2"
    result = normalizer.normalize(text)
    assert "Word 1 Word 2" in result


def test_duplicate_exact_detection():
    detector = DuplicateDetector()
    text = "This is a test document."
    
    is_dup1, orig1 = detector.check_duplicate("doc1", text)
    is_dup2, orig2 = detector.check_duplicate("doc2", text)
    
    assert not is_dup1
    assert is_dup2
    assert orig2 == "doc1"


def test_duplicate_unique_documents():
    detector = DuplicateDetector()
    
    is_dup1, _ = detector.check_duplicate("doc1", "Text A")
    is_dup2, _ = detector.check_duplicate("doc2", "Text B")
    
    assert not is_dup1
    assert not is_dup2


def test_duplicate_case_insensitive():
    detector = DuplicateDetector()
    
    is_dup1, _ = detector.check_duplicate("doc1", "Test Document")
    is_dup2, _ = detector.check_duplicate("doc2", "test document")
    
    assert is_dup2


def test_document_cleaner_full_pipeline():
    cleaner = DocumentCleaner()
    text = "<p>Revenue increased by <b>12%</b>.</p>"
    
    cleaned, metadata = cleaner.clean_document("test_doc", text)
    
    assert metadata["status"] == "success"
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "12%" in cleaned


def test_document_cleaner_metadata():
    cleaner = DocumentCleaner()
    text = "Test document"
    
    cleaned, metadata = cleaner.clean_document("test_doc", text)
    
    assert "document_id" in metadata
    assert "original_length" in metadata
    assert "cleaned_length" in metadata
    assert "steps_completed" in metadata


def test_document_cleaner_duplicate_check():
    cleaner = DocumentCleaner()
    text = "Test document"
    
    is_dup1, _ = cleaner.check_duplicate("doc1", text)
    is_dup2, _ = cleaner.check_duplicate("doc2", text)
    
    assert not is_dup1
    assert is_dup2


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Preprocessing Tests")
    print("=" * 60)
    
    tests = [
        ("Encoding: HTML entity decoding", test_encoding_html_entity_decoding),
        ("Encoding: Unicode normalization", test_encoding_unicode_normalization),
        ("Encoding: Special char normalization", test_encoding_special_char_normalization),
        ("HTML: Simple tag removal", test_html_simple_tag_removal),
        ("HTML: Script removal", test_html_script_removal),
        ("HTML: Comment removal", test_html_comment_removal),
        ("XBRL: Context removal", test_xbrl_context_removal),
        ("XBRL: Namespace tag removal", test_xbrl_namespace_tag_removal),
        ("Table: Simple extraction", test_table_simple_extraction),
        ("Table: Multiple rows", test_table_multiple_rows),
        ("Structure: PART heading", test_structure_part_heading),
        ("Structure: ITEM heading", test_structure_item_heading),
        ("Whitespace: Excessive blank lines", test_whitespace_excessive_blank_lines),
        ("Whitespace: Trailing whitespace", test_whitespace_trailing),
        ("Whitespace: Multiple spaces", test_whitespace_multiple_spaces),
        ("Duplicate: Exact detection", test_duplicate_exact_detection),
        ("Duplicate: Unique documents", test_duplicate_unique_documents),
        ("Duplicate: Case insensitive", test_duplicate_case_insensitive),
        ("Document Cleaner: Full pipeline", test_document_cleaner_full_pipeline),
        ("Document Cleaner: Metadata", test_document_cleaner_metadata),
        ("Document Cleaner: Duplicate check", test_document_cleaner_duplicate_check),
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
