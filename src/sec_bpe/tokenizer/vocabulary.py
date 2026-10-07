"""Vocabulary management for BPE tokenizer."""

import base64
from typing import Dict, List, Optional


class Vocabulary:
    """Manage BPE vocabulary with byte-level tokens."""

    def __init__(self, special_tokens: List[str]):
        """Initialize vocabulary.

        Args:
            special_tokens: List of special token strings.
        """
        self.special_tokens = special_tokens
        self.token_to_id: Dict[bytes, int] = {}
        self.id_to_token: Dict[int, bytes] = {}
        self.token_type: Dict[int, str] = {}  # "special" or "bpe"

        # Assign special token IDs starting from 0
        for i, token in enumerate(special_tokens):
            token_bytes = token.encode("utf-8")
            self.token_to_id[token_bytes] = i
            self.id_to_token[i] = token_bytes
            self.token_type[i] = "special"

        # Add base byte vocabulary (256 bytes)
        self._add_base_bytes()

    def _add_base_bytes(self) -> None:
        """Add 256 base byte tokens to vocabulary."""
        base_start = len(self.special_tokens)
        for byte_val in range(256):
            token_bytes = bytes([byte_val])
            token_id = base_start + byte_val
            self.token_to_id[token_bytes] = token_id
            self.id_to_token[token_id] = token_bytes
            self.token_type[token_id] = "byte"

    def add_merge_token(self, token_bytes: bytes) -> int:
        """Add a new BPE merge token.

        Args:
            token_bytes: Byte sequence for the new token.

        Returns:
            New token ID.
        """
        if token_bytes in self.token_to_id:
            return self.token_to_id[token_bytes]

        new_id = len(self.id_to_token)
        self.token_to_id[token_bytes] = new_id
        self.id_to_token[new_id] = token_bytes
        self.token_type[new_id] = "bpe"
        return new_id

    def get_id(self, token_bytes: bytes) -> Optional[int]:
        """Get token ID for byte sequence.

        Args:
            token_bytes: Byte sequence.

        Returns:
            Token ID or None if not found.
        """
        return self.token_to_id.get(token_bytes)

    def get_token(self, token_id: int) -> Optional[bytes]:
        """Get byte sequence for token ID.

        Args:
            token_id: Token ID.

        Returns:
            Byte sequence or None if not found.
        """
        return self.id_to_token.get(token_id)

    def get_type(self, token_id: int) -> Optional[str]:
        """Get token type.

        Args:
            token_id: Token ID.

        Returns:
            Token type ("special", "byte", or "bpe") or None.
        """
        return self.token_type.get(token_id)

    def size(self) -> int:
        """Get vocabulary size.

        Returns:
            Number of tokens in vocabulary.
        """
        return len(self.id_to_token)

    def base_byte_start(self) -> int:
        """Get starting ID for base byte tokens.

        Returns:
            First byte token ID.
        """
        return len(self.special_tokens)

    def bpe_start(self) -> int:
        """Get starting ID for BPE merge tokens.

        Returns:
            First BPE token ID.
        """
        return len(self.special_tokens) + 256

    def serialize_token(self, token_id: int) -> str:
        """Serialize token to hex string.

        Args:
            token_id: Token ID.

        Returns:
            Hex string representation.
        """
        token_bytes = self.id_to_token[token_id]
        return token_bytes.hex()

    def deserialize_token(self, hex_str: str) -> bytes:
        """Deserialize hex string to bytes.

        Args:
            hex_str: Hex string.

        Returns:
            Byte sequence.
        """
        return bytes.fromhex(hex_str)

    def to_dict(self) -> Dict:
        """Convert vocabulary to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        result = {}
        for token_id in sorted(self.id_to_token.keys()):
            token_bytes = self.id_to_token[token_id]
            result[str(token_id)] = {
                "bytes": token_bytes.hex(),
                "type": self.token_type[token_id],
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "Vocabulary":
        """Load vocabulary from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Vocabulary instance.
        """
        # Extract special tokens
        special_tokens = []
        for token_id in sorted(int(k) for k in data.keys()):
            if data[str(token_id)]["type"] == "special":
                token_bytes = bytes.fromhex(data[str(token_id)]["bytes"])
                special_tokens.append(token_bytes.decode("utf-8"))

        vocab = cls(special_tokens)

        # Add BPE tokens
        for token_id in sorted(int(k) for k in data.keys()):
            if data[str(token_id)]["type"] == "bpe":
                token_bytes = bytes.fromhex(data[str(token_id)]["bytes"])
                vocab.add_merge_token(token_bytes)

        return vocab
