"""Byte-level encoding for BPE tokenizer."""

from typing import List, Tuple


class ByteEncoder:
    """Convert text to byte token IDs."""

    def __init__(self, vocabulary, special_tokens):
        """Initialize byte encoder.

        Args:
            vocabulary: Vocabulary instance.
            special_tokens: SpecialTokens instance.
        """
        self.vocabulary = vocabulary
        self.special_tokens = special_tokens
        self.base_byte_start = vocabulary.base_byte_start()

    def encode_text(self, text: str) -> List[int]:
        """Encode text to byte token IDs.

        Args:
            text: Input text.

        Returns:
            List of token IDs.
        """
        # Convert to bytes
        byte_sequence = text.encode("utf-8")

        # Convert each byte to token ID
        token_ids = []
        for byte_val in byte_sequence:
            token_id = self.base_byte_start + byte_val
            token_ids.append(token_id)

        return token_ids

    def encode_document(self, text: str) -> List[int]:
        """Encode document with special tokens.

        Args:
            text: Document text.

        Returns:
            List of token IDs with special tokens.
        """
        # Add document start token
        doc_start_id = self.special_tokens.get_id("<DOCUMENT_START>")

        # Encode content
        content_ids = self.encode_text(text)

        # Add document end token
        doc_end_id = self.special_tokens.get_id("<DOCUMENT_END>")

        return [doc_start_id] + content_ids + [doc_end_id]

    def encode_documents(self, documents: List[str]) -> List[List[int]]:
        """Encode multiple documents.

        Args:
            documents: List of document texts.

        Returns:
            List of token ID lists (one per document).
        """
        return [self.encode_document(doc) for doc in documents]

    def decode_ids(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded text.
        """
        # Filter out special tokens
        byte_values = []
        for token_id in token_ids:
            if self.special_tokens.is_special_id(token_id):
                continue

            # Get byte value from token ID
            if token_id >= self.base_byte_start:
                byte_val = token_id - self.base_byte_start
                byte_values.append(byte_val)

        # Convert bytes to text
        return bytes(byte_values).decode("utf-8")
