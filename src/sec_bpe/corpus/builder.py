"""Corpus builder for combining SEC filings into a single corpus file."""

from pathlib import Path

from sec_bpe.corpus.loader import CorpusLoader
from sec_bpe.corpus.ingestion_config import IngestionConfig


class CorpusBuilder:
    """Build a single corpus file from multiple SEC filing documents."""

    def __init__(
        self,
        input_dir: str | Path,
        output_path: str | Path,
        config: IngestionConfig = None,
    ):
        """Initialize the corpus builder.

        Args:
            input_dir: Directory containing SEC filing documents.
            output_path: Path where the combined corpus file will be written.
            config: Ingestion configuration (optional).
        """
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.config = config or IngestionConfig()
        self.loader = CorpusLoader(input_dir, self.config)

    def build(self) -> int:
        """Build the corpus file from all supported documents.

        Loads all documents from the input directory and combines them
        into a single output file. Each document is wrapped with
        <DOCUMENT_START> and <DOCUMENT_END> markers.

        Returns:
            Number of documents written to the corpus.

        Raises:
            FileNotFoundError: If the input directory does not exist.
            ValueError: If no supported documents are found.
        """
        documents = self.loader.load_all()

        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write documents to corpus file
        with self.output_path.open("w", encoding="utf-8") as f:
            for i, doc in enumerate(documents):
                if i > 0:
                    f.write("\n\n")
                f.write("<DOCUMENT_START>\n")
                f.write(doc)
                f.write("\n<DOCUMENT_END>")

        return len(documents)
