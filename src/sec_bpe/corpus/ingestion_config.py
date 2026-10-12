"""Configuration for Stage 1 corpus ingestion."""

from dataclasses import dataclass


@dataclass
class IngestionConfig:
    """Configuration for SEC filing corpus ingestion.

    Attributes:
        extract_zip: Whether to extract ZIP archives before processing.
        max_extracted_file_size: Maximum size of a single extracted file in bytes (0 = no limit).
        max_total_extracted_size: Maximum total extraction size in bytes (0 = no limit).
        extract_dir: Directory for extracted files (relative to input_dir).
    """

    extract_zip: bool = False
    max_extracted_file_size: int = 100 * 1024 * 1024  # 100 MB default
    max_total_extracted_size: int = 1024 * 1024 * 1024  # 1 GB default
    extract_dir: str = "_extracted"

    def __post_init__(self):
        """Validate configuration."""
        if self.max_extracted_file_size < 0:
            raise ValueError("max_extracted_file_size must be non-negative")
        if self.max_total_extracted_size < 0:
            raise ValueError("max_total_extracted_size must be non-negative")
