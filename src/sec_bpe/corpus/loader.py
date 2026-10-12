"""Corpus loader for SEC filings."""

from pathlib import Path

from sec_bpe.corpus.ingestion_config import IngestionConfig
from sec_bpe.corpus.zip_extractor import ZipExtractor


class CorpusLoader:
    """Load and discover SEC filing documents from a directory."""

    SUPPORTED_EXTENSIONS = {".txt", ".html", ".htm"}

    def __init__(self, input_dir: str | Path, config: IngestionConfig = None):
        """Initialize the corpus loader.

        Args:
            input_dir: Directory containing SEC filing documents.
            config: Ingestion configuration (optional).
        """
        self.input_dir = Path(input_dir)
        self.config = config or IngestionConfig()
        self.extraction_report = None

    def find_documents(self) -> list[Path]:
        """Find all supported documents in the input directory.

        Recursively searches the directory for files with supported extensions.
        Returns paths in deterministic (sorted) order.

        If ZIP extraction is enabled, extracts archives before discovery.

        Returns:
            List of paths to supported documents.

        Raises:
            FileNotFoundError: If the input directory does not exist.
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_dir}")

        # Extract ZIP archives if enabled
        if self.config.extract_zip:
            extractor = ZipExtractor(self.config)
            self.extraction_report = extractor.extract_archives(self.input_dir)
        else:
            # Initialize report even when disabled
            self.extraction_report = {
                "enabled": False,
                "archives_discovered": 0,
                "archives_extracted": 0,
                "archives_skipped": 0,
                "archives_failed": 0,
                "files_extracted": 0,
                "errors": [],
                "extract_dir": str(self.input_dir / self.config.extract_dir),
            }

        # Find documents
        documents = []
        for path in self.input_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                documents.append(path)

        return sorted(documents)

    def load_document(self, path: Path) -> str:
        """Load a single document as raw text.

        Args:
            path: Path to the document file.

        Returns:
            Raw document contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        return path.read_text(encoding="utf-8", errors="replace")

    def load_all(self) -> list[str]:
        """Load all supported documents from the input directory.

        Returns:
            List of document contents as strings.

        Raises:
            ValueError: If no supported documents are found.
        """
        documents = self.find_documents()

        if not documents:
            raise ValueError(f"No supported documents found in {self.input_dir}")

        return [self.load_document(path) for path in documents]
