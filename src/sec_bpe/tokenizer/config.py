"""BPE tokenizer configuration."""

from dataclasses import dataclass, field


@dataclass
class BPEConfig:
    """Configuration for BPE tokenizer training.

    Attributes:
        vocab_size: Target vocabulary size (including base bytes and special tokens)
        min_frequency: Minimum pair frequency to learn a merge
        special_tokens: List of special token strings
        encoding: Text encoding (utf-8)
        trainer: Trainer type ("reference" or "optimized")
    """

    vocab_size: int = 8192
    min_frequency: int = 2
    special_tokens: list[str] = field(
        default_factory=lambda: ["<DOCUMENT_START>", "<DOCUMENT_END>", "<UNK>"]
    )
    encoding: str = "utf-8"
    trainer: str = "reference"

    def __post_init__(self):
        """Validate configuration."""
        if self.vocab_size < 256 + len(self.special_tokens):
            raise ValueError(
                f"vocab_size must be at least 256 + len(special_tokens) = "
                f"{256 + len(self.special_tokens)}, got {self.vocab_size}"
            )
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        if self.encoding not in ["utf-8"]:
            raise ValueError("Only utf-8 encoding is supported")
        if self.trainer not in ["reference", "optimized"]:
            raise ValueError("trainer must be 'reference' or 'optimized'")
