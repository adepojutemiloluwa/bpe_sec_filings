"""Main BPE tokenizer API."""

from pathlib import Path
from typing import List

from sec_bpe.tokenizer.config import BPEConfig
from sec_bpe.tokenizer.special_tokens import SpecialTokens
from sec_bpe.tokenizer.vocabulary import Vocabulary
from sec_bpe.tokenizer.merge import MergeTable
from sec_bpe.tokenizer.encoder import BPEEncoder
from sec_bpe.tokenizer.decoder import BPEDecoder


class BPETokenizer:
    """Byte-level BPE tokenizer for SEC filings."""

    def __init__(
        self,
        config: BPEConfig = None,
        vocabulary: Vocabulary = None,
        merge_table: MergeTable = None,
    ):
        """Initialize tokenizer.

        Args:
            config: BPE configuration.
            vocabulary: Vocabulary (if loading trained tokenizer).
            merge_table: Merge table (if loading trained tokenizer).
        """
        self.config = config or BPEConfig()

        if vocabulary is None or merge_table is None:
            # Initialize new tokenizer
            self.special_tokens = SpecialTokens(self.config.special_tokens)
            self.vocabulary = Vocabulary(self.config.special_tokens)
            self.merge_table = MergeTable()
        else:
            # Load existing tokenizer
            self.vocabulary = vocabulary
            self.merge_table = merge_table
            self.special_tokens = SpecialTokens(self.config.special_tokens)

        # Initialize encoder and decoder
        self.encoder = BPEEncoder(self.vocabulary, self.special_tokens, self.merge_table)
        self.decoder = BPEDecoder(self.vocabulary, self.special_tokens)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocabulary.size()

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text.

        Returns:
            List of token IDs.
        """
        return self.encoder.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded text.
        """
        return self.decoder.decode(token_ids)

    def tokenize(self, text: str) -> List[dict]:
        """Encode text and return token information.

        Args:
            text: Input text.

        Returns:
            List of token dictionaries with id and bytes.
        """
        return self.encoder.tokenize(text)

    def token_to_id(self, token_bytes: bytes) -> int:
        """Get token ID for byte sequence.

        Args:
            token_bytes: Byte sequence.

        Returns:
            Token ID or None if not found.
        """
        return self.vocabulary.get_id(token_bytes)

    def id_to_token(self, token_id: int) -> bytes:
        """Get byte sequence for token ID.

        Args:
            token_id: Token ID.

        Returns:
            Byte sequence or None if not found.
        """
        return self.vocabulary.get_token(token_id)

    def get_merge(self, rank: int):
        """Get merge by rank.

        Args:
            rank: Merge rank.

        Returns:
            Merge instance.
        """
        return self.merge_table.get_merge(rank)

    def inspect_merges(self, n: int = 10) -> List[dict]:
        """Inspect the first n merges.

        Args:
            n: Number of merges to inspect.

        Returns:
            List of merge dictionaries.
        """
        merges = []
        for rank in range(min(n, self.merge_table.size())):
            merge = self.merge_table.get_merge(rank)
            left_bytes = self.vocabulary.get_token(merge.left_id)
            right_bytes = self.vocabulary.get_token(merge.right_id)
            new_bytes = self.vocabulary.get_token(merge.new_token_id)

            merges.append({
                "rank": merge.rank,
                "left": left_bytes.hex() if left_bytes else None,
                "right": right_bytes.hex() if right_bytes else None,
                "new_token": new_bytes.hex() if new_bytes else None,
                "frequency": merge.frequency,
            })

        return merges

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load tokenizer from directory.

        Args:
            path: Path to tokenizer directory.

        Returns:
            BPETokenizer instance.
        """
        from sec_bpe.tokenizer.serializer import TokenizerSerializer

        serializer = TokenizerSerializer()
        return serializer.load(path)

    def save(self, path: Path) -> None:
        """Save tokenizer to directory.

        Args:
            path: Path to tokenizer directory.
        """
        from sec_bpe.tokenizer.serializer import TokenizerSerializer

        serializer = TokenizerSerializer()
        serializer.save(self, path)
