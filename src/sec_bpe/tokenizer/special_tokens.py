"""Special token handling for BPE tokenizer."""

from typing import Dict, List


class SpecialTokens:
    """Manage special tokens for the BPE tokenizer."""

    def __init__(self, special_tokens: List[str]):
        """Initialize special tokens.

        Args:
            special_tokens: List of special token strings.
        """
        self.special_tokens = special_tokens
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Assign IDs starting from 0
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def get_id(self, token: str) -> int:
        """Get the ID for a special token.

        Args:
            token: Special token string.

        Returns:
            Token ID.

        Raises:
            KeyError: If token is not a special token.
        """
        if token not in self.token_to_id:
            raise KeyError(f"'{token}' is not a special token")
        return self.token_to_id[token]

    def get_token(self, token_id: int) -> str:
        """Get the token string for an ID.

        Args:
            token_id: Token ID.

        Returns:
            Token string.

        Raises:
            KeyError: If ID is not a special token.
        """
        if token_id not in self.id_to_token:
            raise KeyError(f"ID {token_id} is not a special token")
        return self.id_to_token[token_id]

    def is_special(self, token: str) -> bool:
        """Check if a token is a special token.

        Args:
            token: Token string.

        Returns:
            True if special token.
        """
        return token in self.token_to_id

    def is_special_id(self, token_id: int) -> bool:
        """Check if an ID is a special token ID.

        Args:
            token_id: Token ID.

        Returns:
            True if special token ID.
        """
        return token_id in self.id_to_token

    def count(self) -> int:
        """Get the number of special tokens.

        Returns:
            Number of special tokens.
        """
        return len(self.special_tokens)

    def max_id(self) -> int:
        """Get the maximum special token ID.

        Returns:
            Maximum ID.
        """
        return max(self.id_to_token.keys()) if self.id_to_token else -1

    def to_list(self) -> List[str]:
        """Get special tokens as a list.

        Returns:
            List of special token strings.
        """
        return self.special_tokens.copy()
