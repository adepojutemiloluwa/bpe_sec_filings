"""Tokenizer serialization for saving and loading."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from sec_bpe.tokenizer.tokenizer import BPETokenizer


class TokenizerSerializer:
    """Serialize and deserialize BPE tokenizer."""

    def save(self, tokenizer: BPETokenizer, path: Path) -> None:
        """Save tokenizer to directory.

        Args:
            tokenizer: BPETokenizer instance.
            path: Path to tokenizer directory.
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        vocab_path = path / "vocab.json"
        vocab_data = tokenizer.vocabulary.to_dict()
        vocab_path.write_text(json.dumps(vocab_data, indent=2), encoding="utf-8")

        # Save merges
        merges_path = path / "merges.json"
        merges_data = tokenizer.merge_table.to_list()
        merges_path.write_text(json.dumps(merges_data, indent=2), encoding="utf-8")

        # Save configuration
        config_path = path / "config.json"
        config_data = {
            "algorithm": "byte_level_bpe",
            "vocab_size": tokenizer.config.vocab_size,
            "min_frequency": tokenizer.config.min_frequency,
            "encoding": tokenizer.config.encoding,
            "trainer": tokenizer.config.trainer,
            "special_tokens": tokenizer.config.special_tokens,
        }
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        # Save tokenizer metadata
        tokenizer_path = path / "tokenizer.json"
        tokenizer_data = {
            "vocab_size": tokenizer.vocab_size,
            "merge_count": tokenizer.merge_table.size(),
            "special_token_count": tokenizer.special_tokens.count(),
            "base_byte_start": tokenizer.vocabulary.base_byte_start(),
            "bpe_start": tokenizer.vocabulary.bpe_start(),
        }
        tokenizer_path.write_text(json.dumps(tokenizer_data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> BPETokenizer:
        """Load tokenizer from directory.

        Args:
            path: Path to tokenizer directory.

        Returns:
            BPETokenizer instance.
        """
        # Load configuration
        config_path = path / "config.json"
        config_data = json.loads(config_path.read_text(encoding="utf-8"))

        from sec_bpe.tokenizer.config import BPEConfig
        config = BPEConfig(
            vocab_size=config_data["vocab_size"],
            min_frequency=config_data["min_frequency"],
            special_tokens=config_data["special_tokens"],
            encoding=config_data["encoding"],
            trainer=config_data["trainer"],
        )

        # Load vocabulary
        vocab_path = path / "vocab.json"
        vocab_data = json.loads(vocab_path.read_text(encoding="utf-8"))
        from sec_bpe.tokenizer.vocabulary import Vocabulary
        vocabulary = Vocabulary.from_dict(vocab_data)

        # Load merges
        merges_path = path / "merges.json"
        merges_data = json.loads(merges_path.read_text(encoding="utf-8"))
        from sec_bpe.tokenizer.merge import MergeTable
        merge_table = MergeTable.from_list(merges_data)

        # Create tokenizer
        return BPETokenizer(config=config, vocabulary=vocabulary, merge_table=merge_table)

    def save_training_metadata(
        self,
        path: Path,
        corpus_path: Path,
        initial_token_count: int,
        final_token_count: int,
    ) -> None:
        """Save training metadata including corpus hash.

        Args:
            path: Path to tokenizer directory.
            corpus_path: Path to training corpus。
            initial_token_count: Initial token count.
            final_token_count: Final token count.
        """
        # Calculate corpus hash
        corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

        metadata = {
            "corpus_path": str(corpus_path),
            "corpus_hash": corpus_hash,
            "initial_token_count": initial_token_count,
            "final_token_count": final_token_count,
            "compression_ratio": initial_token_count / final_token_count if final_token_count > 0 else 0,
            "created_at": datetime.utcnow().isoformat(),
        }

        metadata_path = path / "training_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
