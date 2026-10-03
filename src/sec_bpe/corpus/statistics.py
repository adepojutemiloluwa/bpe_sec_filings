"""Generate corpus statistics for BPE training."""

import re
from collections import Counter, defaultdict
from typing import Dict, List


class CorpusStatistics:
    """Generate comprehensive corpus statistics."""

    def __init__(self):
        """Initialize the statistics generator."""
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.character_counts = Counter()
        self.byte_counts = Counter()
        self.document_lengths = []
        self.company_counts = Counter()
        self.financial_patterns = Counter()
        self.total_characters = 0
        self.total_bytes = 0

    def analyze_documents(self, documents: List[Dict]) -> Dict:
        """Analyze a list of documents.

        Args:
            documents: List of document dictionaries.

        Returns:
            Dictionary with all statistics.
        """
        self.reset()

        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]

            # Document length
            self.document_lengths.append(len(text))

            # Character counts
            for char in text:
                self.character_counts[char] += 1
            self.total_characters += len(text)

            # Byte counts
            for byte in text.encode("utf-8"):
                self.byte_counts[byte] += 1
            self.total_bytes += len(text.encode("utf-8"))

            # Company counts
            ticker = metadata.get("ticker", "UNKNOWN")
            self.company_counts[ticker] += 1

            # Financial patterns
            self._count_financial_patterns(text)

        return self.generate_report()

    def _count_financial_patterns(self, text: str) -> None:
        """Count financial patterns in text.

        Args:
            text: Document text.
        """
        # Percent signs
        self.financial_patterns["%"] += text.count("%")

        # Dollar signs
        self.financial_patterns["$"] += text.count("$")

        # Form types
        self.financial_patterns["10-K"] += text.count("10-K")
        self.financial_patterns["10-Q"] += text.count("10-Q")

        # Structure markers
        self.financial_patterns["ITEM"] += text.count("ITEM")
        self.financial_patterns["PART"] += text.count("PART")

        # Common financial terms
        financial_terms = [
            "million", "billion", "revenue", "assets", "liabilities",
            "income", "cash flow", "net income", "operating", "quarter",
        ]
        for term in financial_terms:
            self.financial_patterns[term] += text.lower().count(term.lower())

    def generate_report(self) -> Dict:
        """Generate statistics report.

        Returns:
            Dictionary with all statistics.
        """
        report = {
            "document_statistics": self._document_statistics(),
            "character_statistics": self._character_statistics(),
            "byte_statistics": self._byte_statistics(),
            "document_length_statistics": self._document_length_statistics(),
            "financial_pattern_statistics": self._financial_pattern_statistics(),
            "company_statistics": self._company_statistics(),
            "vocabulary_candidates": self._vocabulary_candidates(),
        }

        return report

    def _document_statistics(self) -> Dict:
        """Generate document-level statistics.

        Returns:
            Dictionary with document statistics.
        """
        return {
            "number_of_documents": len(self.document_lengths),
            "number_of_companies": len(self.company_counts),
            "number_of_filings_per_company": dict(self.company_counts.most_common(10)),
        }

    def _character_statistics(self) -> Dict:
        """Generate character-level statistics.

        Returns:
            Dictionary with character statistics.
        """
        total = self.total_characters
        unique = len(self.character_counts)

        # Character frequency
        char_freq = dict(self.character_counts.most_common(100))

        # Character frequency percentage
        char_freq_pct = {
            char: (count / total * 100) if total > 0 else 0
            for char, count in char_freq.items()
        }

        return {
            "total_characters": total,
            "unique_characters": unique,
            "character_frequency": char_freq,
            "character_frequency_percentage": char_freq_pct,
        }

    def _byte_statistics(self) -> Dict:
        """Generate byte-level statistics.

        Returns:
            Dictionary with byte statistics.
        """
        total = self.total_bytes
        unique = len(self.byte_counts)

        # Byte frequency
        byte_freq = dict(self.byte_counts.most_common(100))

        # Byte frequency percentage
        byte_freq_pct = {
            byte: (count / total * 100) if total > 0 else 0
            for byte, count in byte_freq.items()
        }

        return {
            "total_bytes": total,
            "unique_bytes": unique,
            "byte_frequency": byte_freq,
            "byte_frequency_percentage": byte_freq_pct,
        }

    def _document_length_statistics(self) -> Dict:
        """Generate document length statistics.

        Returns:
            Dictionary with length statistics.
        """
        if not self.document_lengths:
            return {
                "minimum_document_length": 0,
                "maximum_document_length": 0,
                "mean_document_length": 0,
                "median_document_length": 0,
                "percentiles": {},
            }

        sorted_lengths = sorted(self.document_lengths)
        n = len(sorted_lengths)

        return {
            "minimum_document_length": min(sorted_lengths),
            "maximum_document_length": max(sorted_lengths),
            "mean_document_length": sum(sorted_lengths) / n,
            "median_document_length": sorted_lengths[n // 2],
            "percentiles": {
                "25th": sorted_lengths[int(n * 0.25)],
                "50th": sorted_lengths[int(n * 0.50)],
                "75th": sorted_lengths[int(n * 0.75)],
                "90th": sorted_lengths[int(n * 0.90)],
                "95th": sorted_lengths[int(n * 0.95)],
                "99th": sorted_lengths[int(n * 0.99)],
            },
        }

    def _financial_pattern_statistics(self) -> Dict:
        """Generate financial pattern statistics.

        Returns:
            Dictionary with pattern statistics.
        """
        return dict(self.financial_patterns)

    def _company_statistics(self) -> Dict:
        """Generate company-level statistics.

        Returns:
            Dictionary with company statistics.
        """
        return {
            "total_companies": len(self.company_counts),
            "companies_with_multiple_filings": sum(1 for count in self.company_counts.values() if count > 1),
            "top_companies_by_filing_count": dict(self.company_counts.most_common(20)),
        }

    def _vocabulary_candidates(self) -> Dict:
        """Generate vocabulary candidate analysis.

        Returns:
            Dictionary with vocabulary candidates.
        """
        # Unique characters
        unique_chars = set(self.character_counts.keys())

        # Unique bytes
        unique_bytes = set(self.byte_counts.keys())

        # Whitespace patterns
        whitespace_patterns = set()
        for char in unique_chars:
            if char.isspace():
                whitespace_patterns.add(char)

        # Common punctuation sequences
        punctuation = set(".,:;%$-/()\"'&+=*")
        common_punctuation = punctuation & unique_chars

        # Numeric patterns
        numeric_chars = set("0123456789.,%")
        numeric_in_corpus = numeric_chars & unique_chars

        return {
            "unique_characters": len(unique_chars),
            "unique_bytes": len(unique_bytes),
            "unique_whitespace_patterns": len(whitespace_patterns),
            "whitespace_patterns": list(whitespace_patterns),
            "common_punctuation": list(common_punctuation),
            "numeric_characters": list(numeric_in_corpus),
        }

    def analyze_rare_characters(self, threshold: int = 10) -> List[Dict]:
        """Analyze rare characters.

        Args:
            threshold: Maximum frequency to consider "rare".

        Returns:
            List of rare character dictionaries.
        """
        rare_chars = []
        for char, count in self.character_counts.items():
            if count <= threshold:
                rare_chars.append({
                    "character": char,
                    "frequency": count,
                    "percentage": (count / self.total_characters * 100) if self.total_characters > 0 else 0,
                    "unicode_code_point": f"U+{ord(char):04X}",
                })

        # Sort by frequency
        rare_chars.sort(key=lambda x: x["frequency"])

        return rare_chars

    def analyze_split_statistics(self, splits: Dict[str, List[Dict]]) -> Dict:
        """Analyze statistics per split.

        Args:
            splits: Dictionary with train/validation/test document lists.

        Returns:
            Dictionary with per-split statistics.
        """
        split_stats = {}

        for split_name, documents in splits.items():
            if not documents:
                continue

            # Count documents
            n_docs = len(documents)

            # Count characters
            total_chars = sum(len(doc["text"]) for doc in documents)

            # Count bytes
            total_bytes = sum(len(doc["text"].encode("utf-8")) for doc in documents)

            # Count companies
            companies = set(doc["metadata"].get("ticker", "UNKNOWN") for doc in documents)

            split_stats[split_name] = {
                "documents": n_docs,
                "characters": total_chars,
                "bytes": total_bytes,
                "companies": len(companies),
            }

        return split_stats
