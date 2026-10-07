"""Tokenizer artifact validation."""

import json
from pathlib import Path
from typing import Dict, List


class TokenizerValidator:
    """Validate saved tokenizer artifacts."""

    def __init__(self, path: Path):
        """Initialize validator.

        Args:
            path: Path to tokenizer directory.
        """
        self.path = path
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """Run all validation checks.

        Returns:
            True if validation passes, False otherwise.
        """
        self.errors = []
        self.warnings = []

        # Check required files
        self._check_required_files()

        # Load and validate config
        config = self._load_config()
        if config:
            self._validate_config(config)

        # Load and validate vocabulary
        vocab = self._load_vocabulary()
        if vocab:
            self._validate_vocabulary(vocab)

        # Load and validate merges
        merges = self._load_merges()
        if merges:
            self._validate_merges(merges)

        # Cross-validate
        if vocab and merges and config:
            self._validate_cross_consistency(vocab, merges, config)

        return len(self.errors) == 0

    def _check_required_files(self) -> None:
        """Check that required files exist."""
        required_files = ["vocab.json", "merges.json", "config.json", "tokenizer.json"]

        for filename in required_files:
            file_path = self.path / filename
            if not file_path.exists():
                self.errors.append(f"Missing required file: {filename}")

    def _load_config(self) -> Dict:
        """Load configuration file.

        Returns:
            Configuration dictionary or None if error.
        """
        config_path = self.path / "config.json"
        if not config_path.exists():
            return None

        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.errors.append(f"Failed to load config.json: {e}")
            return None

    def _validate_config(self, config: Dict) -> None:
        """Validate configuration.

        Args:
            config: Configuration dictionary.
        """
        required_fields = ["algorithm", "vocab_size", "min_frequency", "special_tokens"]

        for field in required_fields:
            if field not in config:
                self.errors.append(f"Config missing field: {field}")

        if config.get("algorithm") != "byte_level_bpe":
            self.errors.append(f"Invalid algorithm: {config.get('algorithm')}")

        if config.get("vocab_size", 0) < 256:
            self.errors.append(f"vocab_size too small: {config.get('vocab_size')}")

        if not isinstance(config.get("special_tokens"), list):
            self.errors.append("special_tokens must be a list")

    def _load_vocabulary(self) -> Dict:
        """Load vocabulary file.

        Returns:
            Vocabulary dictionary or None if error.
        """
        vocab_path = self.path / "vocab.json"
        if not vocab_path.exists():
            return None

        try:
            return json.loads(vocab_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.errors.append(f"Failed to load vocab.json: {e}")
            return None

    def _validate_vocabulary(self, vocab: Dict) -> None:
        """Validate vocabulary.

        Args:
            vocab: Vocabulary dictionary.
        """
        # Check for duplicate IDs
        ids = [int(k) for k in vocab.keys()]
        if len(ids) != len(set(ids)):
            self.errors.append("Vocabulary contains duplicate IDs")

        # Check IDs are contiguous
        sorted_ids = sorted(ids)
        for i in range(1, len(sorted_ids)):
            if sorted_ids[i] != sorted_ids[i - 1] + 1:
                self.warnings.append(f"Vocabulary IDs are not contiguous: gap at {sorted_ids[i - 1]}")

        # Check each token entry
        for token_id_str, token_data in vocab.items():
            if "bytes" not in token_data:
                self.errors.append(f"Token {token_id_str} missing 'bytes' field")
            if "type" not in token_data:
                self.errors.append(f"Token {token_id_str} missing 'type' field")
            if token_data.get("type") not in ["special", "byte", "bpe"]:
                self.errors.append(f"Token {token_id_str} has invalid type: {token_data.get('type')}")

            # Validate hex string
            try:
                bytes.fromhex(token_data["bytes"])
            except ValueError:
                self.errors.append(f"Token {token_id_str} has invalid hex bytes: {token_data['bytes']}")

    def _load_merges(self) -> List:
        """Load merges file.

        Returns:
            Merges list or None if error.
        """
        merges_path = self.path / "merges.json"
        if not merges_path.exists():
            return None

        try:
            return json.loads(merges_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.errors.append(f"Failed to load merges.json: {e}")
            return None

    def _validate_merges(self, merges: List) -> None:
        """Validate merges.

        Args:
            merges: Merges list.
        """
        # Check for duplicate ranks
        ranks = [m.get("rank") for m in merges]
        if len(ranks) != len(set(ranks)):
            self.errors.append("Merges contain duplicate ranks")

        # Check ranks are ordered
        sorted_ranks = sorted(ranks)
        if ranks != sorted_ranks:
            self.errors.append("Merges are not ordered by rank")

        # Check each merge entry
        required_fields = ["rank", "left", "right", "new_token_id", "frequency"]

        for i, merge in enumerate(merges):
            for field in required_fields:
                if field not in merge:
                    self.errors.append(f"Merge {i} missing field: {field}")

            # Check rank matches position
            if merge.get("rank") != i:
                self.errors.append(f"Merge {i} has incorrect rank: {merge.get('rank')}")

            # Check frequency is non-negative
            if merge.get("frequency", 0) < 0:
                self.errors.append(f"Merge {i} has negative frequency")

    def _validate_cross_consistency(
        self,
        vocab: Dict,
        merges: List,
        config: Dict,
    ) -> None:
        """Validate cross-consistency between files.

        Args:
            vocab: Vocabulary dictionary.
            merges: Merges list.
            config: Configuration dictionary.
        """
        # Check vocab size matches config
        vocab_size = len(vocab)
        config_vocab_size = config.get("vocab_size")
        if vocab_size != config_vocab_size:
            self.errors.append(
                f"Vocabulary size mismatch: vocab has {vocab_size}, config expects {config_vocab_size}"
            )

        # Check special token count matches config
        special_count = sum(1 for v in vocab.values() if v.get("type") == "special")
        config_special_count = len(config.get("special_tokens", []))
        if special_count != config_special_count:
            self.errors.append(
                f"Special token count mismatch: vocab has {special_count}, config expects {config_special_count}"
            )

        # Check all merge references exist in vocabulary
        vocab_ids = set(int(k) for k in vocab.keys())
        for merge in merges:
            left_id = merge.get("left")
            right_id = merge.get("right")
            new_id = merge.get("new_token_id")

            if left_id not in vocab_ids:
                self.errors.append(f"Merge references non-existent left token ID: {left_id}")
            if right_id not in vocab_ids:
                self.errors.append(f"Merge references non-existent right token ID: {right_id}")
            if new_id not in vocab_ids:
                self.errors.append(f"Merge references non-existent new token ID: {new_id}")

    def get_report(self) -> Dict:
        """Get validation report.

        Returns:
            Validation report dictionary.
        """
        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
        }
