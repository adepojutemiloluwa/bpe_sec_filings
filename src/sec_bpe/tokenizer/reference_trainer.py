"""Reference BPE trainer implementation."""

import logging
from pathlib import Path
from typing import Callable, List, Tuple

from sec_bpe.tokenizer.config import BPEConfig
from sec_bpe.tokenizer.special_tokens import SpecialTokens
from sec_bpe.tokenizer.vocabulary import Vocabulary
from sec_bpe.tokenizer.byte_encoder import ByteEncoder
from sec_bpe.tokenizer.pair_counter import PairCounter
from sec_bpe.tokenizer.merge import MergeTable


class ReferenceBPETrainer:
    """Reference implementation of BPE training.

    This is a simple, correct implementation optimized for clarity
    rather than performance. It implements the standard BPE algorithm:
    1. Convert text to byte token IDs
    2. Count adjacent token pairs
    3. Select most frequent pair
    4. Create new token for that pair
    5. Replace all occurrences
    6. Repeat until vocab size reached
    """

    def __init__(self, config: BPEConfig):
        """Initialize the reference trainer.

        Args:
            config: BPE configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.special_tokens = SpecialTokens(config.special_tokens)
        self.vocabulary = Vocabulary(config.special_tokens)
        self.byte_encoder = ByteEncoder(self.vocabulary, self.special_tokens)
        self.pair_counter = PairCounter(self.special_tokens)
        self.merge_table = MergeTable()

        # Training state
        self.documents: List[List[int]] = []
        self.initial_token_count = 0
        self.final_token_count = 0

    def load_corpus(self, corpus_path: Path) -> None:
        """Load training corpus.

        Args:
            corpus_path: Path to training corpus file.
        """
        self.logger.info(f"Loading corpus from {corpus_path}")

        with corpus_path.open("r", encoding="utf-8") as f:
            corpus_text = f.read()

        # Split by document markers
        documents_text = corpus_text.split("<DOCUMENT_START>")

        for doc_text in documents_text:
            if not doc_text.strip():
                continue

            # Extract content between markers
            if "<DOCUMENT_END>" in doc_text:
                content, _ = doc_text.split("<DOCUMENT_END>", 1)
                content = content.strip()
            else:
                continue

            # Skip very short documents
            if len(content) < 100:
                continue

            # Encode document
            token_ids = self.byte_encoder.encode_document(content)
            self.documents.append(token_ids)

        self.logger.info(f"Loaded {len(self.documents)} documents")

    def train(self) -> Tuple[Vocabulary, MergeTable]:
        """Train BPE tokenizer.

        Returns:
            Tuple of (vocabulary, merge_table).
        """
        self.logger.info("Starting BPE training")
        self.logger.info(f"Target vocabulary size: {self.config.vocab_size}")
        self.logger.info(f"Minimum frequency: {self.config.min_frequency}")

        # Calculate initial token count
        self.initial_token_count = sum(len(doc) for doc in self.documents)
        self.logger.info(f"Initial token count: {self.initial_token_count}")

        # Calculate max merges
        base_vocab_size = self.vocabulary.size()
        max_merges = self.config.vocab_size - base_vocab_size
        self.logger.info(f"Maximum merges to learn: {max_merges}")

        # Training loop
        merge_count = 0
        progress_callback = self._get_progress_callback()

        while merge_count < max_merges:
            # Count pairs
            pair_counts = self.pair_counter.count_pairs_documents(self.documents)

            if not pair_counts:
                self.logger.info("No more pairs to merge")
                break

            # Select most frequent pair
            left_id, right_id, frequency = self.pair_counter.select_most_frequent(
                pair_counts
            )

            # Check minimum frequency
            if frequency < self.config.min_frequency:
                self.logger.info(
                    f"Highest pair frequency ({frequency}) below minimum "
                    f"({self.config.min_frequency}), stopping"
                )
                break

            # Get byte sequences for left and right tokens
            left_bytes = self.vocabulary.get_token(left_id)
            right_bytes = self.vocabulary.get_token(right_id)

            if left_bytes is None or right_bytes is None:
                self.logger.error(f"Cannot find tokens for pair ({left_id}, {right_id})")
                break

            # Create new token
            new_bytes = left_bytes + right_bytes
            new_token_id = self.vocabulary.add_merge_token(new_bytes)

            # Record merge
            self.merge_table.add_merge(left_id, right_id, new_token_id, frequency)

            # Apply merge to all documents
            self._apply_merge(left_id, right_id, new_token_id)

            merge_count += 1

            # Progress reporting
            if merge_count % 100 == 0 or merge_count == max_merges:
                progress_callback(merge_count, max_merges, frequency)

        # Calculate final token count
        self.final_token_count = sum(len(doc) for doc in self.documents)
        compression_ratio = self.initial_token_count / self.final_token_count

        self.logger.info(f"Training complete")
        self.logger.info(f"Merges learned: {merge_count}")
        self.logger.info(f"Final vocabulary size: {self.vocabulary.size()}")
        self.logger.info(f"Final token count: {self.final_token_count}")
        self.logger.info(f"Compression ratio: {compression_ratio:.2f}")

        return self.vocabulary, self.merge_table

    def _apply_merge(self, left_id: int, right_id: int, new_token_id: int) -> None:
        """Apply a merge to all documents.

        Args:
            left_id: Left token ID.
            right_id: Right token ID.
            new_token_id: New token ID.
        """
        for doc_idx, token_ids in enumerate(self.documents):
            new_ids = []
            i = 0

            while i < len(token_ids):
                # Check if current position matches the pair
                if (
                    i + 1 < len(token_ids)
                    and token_ids[i] == left_id
                    and token_ids[i + 1] == right_id
                ):
                    # Replace with new token
                    new_ids.append(new_token_id)
                    i += 2
                else:
                    # Keep current token
                    new_ids.append(token_ids[i])
                    i += 1

            self.documents[doc_idx] = new_ids

    def _get_progress_callback(self) -> Callable:
        """Get progress callback for training.

        Returns:
            Progress callback function.
        """
        def callback(merge_num: int, total: int, frequency: int) -> None:
            percent = (merge_num / total) * 100
            self.logger.info(
                f"Merge {merge_num}/{total} ({percent:.1f}%) - "
                f"frequency: {frequency}"
            )

        return callback
