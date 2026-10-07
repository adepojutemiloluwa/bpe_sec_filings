"""Training statistics for BPE tokenizer."""

import json
from pathlib import Path
from typing import Dict, List


class TrainingStatistics:
    """Track and report BPE training statistics."""

    def __init__(self):
        """Initialize statistics tracker."""
        self.initial_vocab_size = 0
        self.final_vocab_size = 0
        self.merge_count = 0
        self.min_frequency = 0
        self.max_merge_frequency = 0
        self.num_documents = 0
        self.training_corpus_bytes = 0
        self.training_corpus_characters = 0
        self.initial_token_count = 0
        self.final_token_count = 0
        self.merge_frequencies: List[int] = []

    def record_initial_state(
        self,
        vocab_size: int,
        num_documents: int,
        corpus_bytes: int,
        corpus_characters: int,
        token_count: int,
    ) -> None:
        """Record initial training state.

        Args:
            vocab_size: Initial vocabulary size.
            num_documents: Number of training documents.
            corpus_bytes: Total bytes in corpus.
            corpus_characters: Total characters in corpus.
            token_count: Initial token count.
        """
        self.initial_vocab_size = vocab_size
        self.num_documents = num_documents
        self.training_corpus_bytes = corpus_bytes
        self.training_corpus_characters = corpus_characters
        self.initial_token_count = token_count

    def record_merge(self, frequency: int) -> None:
        """Record a merge with its frequency.

        Args:
            frequency: Pair frequency at merge time.
        """
        self.merge_count += 1
        self.merge_frequencies.append(frequency)
        if frequency > self.max_merge_frequency:
            self.max_merge_frequency = frequency

    def record_final_state(
        self,
        vocab_size: int,
        token_count: int,
        min_frequency: int,
    ) -> None:
        """Record final training state.

        Args:
            vocab_size: Final vocabulary size.
            token_count: Final token count.
            min_frequency: Minimum frequency threshold.
        """
        self.final_vocab_size = vocab_size
        self.final_token_count = token_count
        self.min_frequency = min_frequency

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio.

        Returns:
            Compression ratio (initial / final tokens).
        """
        if self.final_token_count == 0:
            return 0.0
        return self.initial_token_count / self.final_token_count

    @property
    def average_bytes_per_token(self) -> float:
        """Calculate average bytes per token.

        Returns:
            Average bytes per token.
        """
        if self.final_token_count == 0:
            return 0.0
        return self.training_corpus_bytes / self.final_token_count

    def to_dict(self) -> Dict:
        """Convert statistics to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "initial_vocab_size": self.initial_vocab_size,
            "final_vocab_size": self.final_vocab_size,
            "merge_count": self.merge_count,
            "min_frequency": self.min_frequency,
            "max_merge_frequency": self.max_merge_frequency,
            "num_documents": self.num_documents,
            "training_corpus_bytes": self.training_corpus_bytes,
            "training_corpus_characters": self.training_corpus_characters,
            "initial_token_count": self.initial_token_count,
            "final_token_count": self.final_token_count,
            "compression_ratio": self.compression_ratio,
            "average_bytes_per_token": self.average_bytes_per_token,
            "merge_frequency_stats": {
                "min": min(self.merge_frequencies) if self.merge_frequencies else 0,
                "max": max(self.merge_frequencies) if self.merge_frequencies else 0,
                "mean": sum(self.merge_frequencies) / len(self.merge_frequencies) if self.merge_frequencies else 0,
            },
        }

    def save(self, path: Path) -> None:
        """Save statistics to JSON file.

        Args:
            path: Path to statistics file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
