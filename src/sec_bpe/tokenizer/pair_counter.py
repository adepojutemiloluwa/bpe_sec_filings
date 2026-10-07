"""Pair frequency counting for BPE training."""

from collections import Counter
from typing import Dict, List, Tuple


class PairCounter:
    """Count adjacent token pair frequencies."""

    def __init__(self, special_tokens):
        """Initialize pair counter.

        Args:
            special_tokens: SpecialTokens instance.
        """
        self.special_tokens = special_tokens

    def count_pairs(self, token_ids: List[int]) -> Counter:
        """Count adjacent pairs in a token sequence.

        Args:
            token_ids: List of token IDs.

        Returns:
            Counter of (left_id, right_id) pairs.
        """
        pairs = Counter()

        for i in range(len(token_ids) - 1):
            left_id = token_ids[i]
            right_id = token_ids[i + 1]

            # Skip pairs involving special tokens
            if self.special_tokens.is_special_id(left_id):
                continue
            if self.special_tokens.is_special_id(right_id):
                continue

            pair = (left_id, right_id)
            pairs[pair] += 1

        return pairs

    def count_pairs_documents(
        self, documents: List[List[int]]
    ) -> Counter:
        """Count pairs across multiple documents.

        Args:
            documents: List of token ID lists (one per document).

        Returns:
            Counter of (left_id, right_id) pairs.
        """
        total_pairs = Counter()

        for token_ids in documents:
            doc_pairs = self.count_pairs(token_ids)
            total_pairs.update(doc_pairs)

        return total_pairs

    def select_most_frequent(self, pair_counts: Counter) -> Tuple[int, int, int]:
        """Select the most frequent pair with deterministic tie-breaking.

        Tie-breaking rule:
        1. Highest frequency wins
        2. If tied, lowest left token ID wins
        3. If still tied, lowest right token ID wins

        Args:
            pair_counts: Counter of pair frequencies.

        Returns:
            Tuple of (left_id, right_id, frequency).

        Raises:
            ValueError: If no pairs available.
        """
        if not pair_counts:
            raise ValueError("No pairs available")

        # Find maximum frequency
        max_freq = max(pair_counts.values())

        # Get all pairs with max frequency
        max_pairs = [
            (left, right)
            for (left, right), freq in pair_counts.items()
            if freq == max_freq
        ]

        # Tie-breaking: lowest left, then lowest right
        best_pair = min(max_pairs, key=lambda p: (p[0], p[1]))

        return best_pair[0], best_pair[1], max_freq
