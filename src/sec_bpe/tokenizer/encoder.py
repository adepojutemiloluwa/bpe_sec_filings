"""BPE encoder for tokenizing text."""

from typing import List, Optional


class BPEEncoder:
    """Encode text using learned BPE merges."""

    def __init__(self, vocabulary, special_tokens, merge_table):
        """Initialize encoder.

        Args:
            vocabulary: Vocabulary instance.
            special_tokens: SpecialTokens instance.
            merge_table: MergeTable instance.
        """
        self.vocabulary = vocabulary
        self.special_tokens = special_tokens
        self.merge_table = merge_table
        self.base_byte_start = vocabulary.base_byte_start()

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using BPE.

        Args:
            text: Input text.

        Returns:
            List of token IDs.
        """
        # Handle special tokens in text
        if self.special_tokens.is_special(text):
            return [self.special_tokens.get_id(text)]

        # Convert to bytes
        byte_sequence = text.encode("utf-8")

        # Convert to initial token IDs
        token_ids = [
            self.base_byte_start + byte_val
            for byte_val in byte_sequence
        ]

        # Apply BPE merges in order of rank
        for rank in range(self.merge_table.size()):
            merge = self.merge_table.get_merge(rank)
            token_ids = self._apply_merge(token_ids, merge.left_id, merge.right_id, merge.new_token_id)

        return token_ids

    def _apply_merge(
        self,
        token_ids: List[int],
        left_id: int,
        right_id: int,
        new_token_id: int,
    ) -> List[int]:
        """Apply a single merge to token sequence.

        Args:
            token_ids: Current token IDs.
            left_id: Left token ID.
            right_id: Right token ID.
            new_token_id: New token ID.

        Returns:
            Updated token IDs.
        """
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

        return new_ids

    def encode_with_special_tokens(self, text: str) -> List[int]:
        """Encode text with document boundary tokens.

        Args:
            text: Input text.

        Returns:
            List of token IDs with special tokens.
        """
        doc_start_id = self.special_tokens.get_id("<DOCUMENT_START>")
        content_ids = self.encode(text)
        doc_end_id = self.special_tokens.get_id("<DOCUMENT_END>")

        return [doc_start_id] + content_ids + [doc_end_id]

    def tokenize(self, text: str) -> List[dict]:
        """Encode text and return token information.

        Args:
            text: Input text.

        Returns:
            List of token dictionaries with id and bytes.
        """
        token_ids = self.encode(text)
        tokens = []

        for token_id in token_ids:
            token_bytes = self.vocabulary.get_token(token_id)
            if token_bytes:
                tokens.append({
                    "id": token_id,
                    "bytes": token_bytes.hex(),
                })

        return tokens
