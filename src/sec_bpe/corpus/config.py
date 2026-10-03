"""Configuration for Stage 3 corpus preparation."""

from dataclasses import dataclass


@dataclass
class CorpusPreparationConfig:
    """Configuration for corpus preparation and splitting.

    Attributes:
        split_strategy: Strategy for splitting ("company", "document", "temporal")
        train_ratio: Proportion of data for training (0.0-1.0)
        validation_ratio: Proportion of data for validation (0.0-1.0)
        test_ratio: Proportion of data for test (0.0-1.0)
        seed: Random seed for reproducibility
        preserve_case: Whether to preserve original case
        preserve_punctuation: Whether to preserve punctuation
        boundary_mode: How to handle document boundaries ("special", "remove")
        min_document_length: Minimum characters for valid document
        max_document_length: Maximum characters (0 = no limit)
    """

    split_strategy: str = "company"
    train_ratio: float = 0.80
    validation_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42
    preserve_case: bool = True
    preserve_punctuation: bool = True
    boundary_mode: str = "special"
    min_document_length: int = 100
    max_document_length: int = 0

    def __post_init__(self):
        """Validate configuration."""
        # Validate split ratios sum to 1.0
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {total:.3f})"
            )

        # Validate split strategy
        valid_strategies = {"company", "document", "temporal"}
        if self.split_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid split strategy: {self.split_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        # Validate boundary mode
        valid_modes = {"special", "remove"}
        if self.boundary_mode not in valid_modes:
            raise ValueError(
                f"Invalid boundary mode: {self.boundary_mode}. "
                f"Must be one of {valid_modes}"
            )

        # Validate ratios are non-negative
        if self.train_ratio < 0 or self.validation_ratio < 0 or self.test_ratio < 0:
            raise ValueError("Split ratios must be non-negative")
