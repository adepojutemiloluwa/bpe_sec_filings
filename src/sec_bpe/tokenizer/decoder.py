"""BPE decoder for reconstructing text from token IDs."""

from typing import List


class BPEDecoder:
    """Decode token IDs back to text."""

    def __init__(self, vocabulary, special_tokens):
        """Initialize decoder.

        Args:
            vocabulary: Vocabulary instance.
            special_tokens: SpecialTokens instance.
        """
        self.vocabulary = vocabulary
        self.special_tokens = special_tokens
        self.base_byte_start = vocabulary.base_byte_start()

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.

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

            # Get byte sequence for token
            token_bytes = self.vocabulary.get_token(token_id)
            if token_bytes:
                byte_values.extend(token_bytes)

        # Convert bytes to text
        return bytes(byte_values).decode("utf-8")

    def decode_with_special_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs including special tokens.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded text with special tokens preserved.
        """
        result_parts = []

        for token_id in token_ids:
            if self.special_tokens.is_special_id(token_id):
                # Add special token as text
                token_str = self.special_tokens.get_token(token_id)
                result_parts.append(token_str)
            else:
                # Get byte sequence
                token_bytes = self.vocabulary.get_token(token_id)
                if token_bytes:
                    result_parts.append(token_bytes.decode("utf-8"))

        return "".join(result_parts)
