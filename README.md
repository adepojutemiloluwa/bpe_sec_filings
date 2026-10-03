# SEC-BPE

A production-quality, reusable Python tokenizer for SEC Form 10-K filings.

## Overview

SEC-BPE implements Byte Pair Encoding (BPE) from scratch, specifically trained on SEC Form 10-K filings. The tokenizer is designed for both serious NLP engineering and as a reusable package for other developers.

## Project Structure

```
sec-bpe/
├── src/sec_bpe/
│   ├── core/          # BPE algorithm implementation
│   ├── preprocessing/ # SEC-specific text normalization
│   ├── corpus/        # Corpus loading and building
│   └── evaluation/    # Benchmarking and metrics
├── scripts/           # Utility scripts
├── experiments/       # Configs, results, and notebooks
└── tests/             # Unit tests
```

## Installation

```bash
pip install -e .
```

## Usage

### Stage 1: Corpus Ingestion

```python
from sec_bpe.corpus.builder import CorpusBuilder

builder = CorpusBuilder(
    input_dir="data/raw",
    output_path="data/processed/sec_corpus.txt",
)

document_count = builder.build()
print(f"Added {document_count} documents")
```

### Stage 2: SEC Filing Cleaning and Normalization

```bash
python scripts/prepare_corpus.py --input-dir data/processed --output-dir data/cleaned
```

Or programmatically:

```python
from sec_bpe.preprocessing.corpus_builder import CleanedCorpusBuilder

builder = CleanedCorpusBuilder(
    input_dir="data/processed",
    output_dir="data/cleaned",
)

stats = builder.build()
print(f"Cleaned {stats['documents_cleaned']} documents")
print(f"Removed {stats['html_elements_removed']} HTML elements")
print(f"Detected {stats['duplicates_detected']} duplicates")
```

## Development Status

- **Stage 1**: SEC filing corpus ingestion ✓
  - Recursive discovery of `.txt`, `.html`, `.htm` files
  - UTF-8 encoding with error handling
  - Document boundary markers (`<DOCUMENT_START>`, `<DOCUMENT_END>`)
  - Deterministic file ordering

- **Stage 2**: SEC preprocessing and normalization ✓
  - Character encoding normalization (HTML entities, Unicode, special chars)
  - HTML/XML removal (scripts, comments, tags)
  - XBRL handling (context, units, namespaces)
  - Table extraction and linear serialization
  - SEC structure preservation (PART, ITEM headings)
  - Whitespace normalization and artifact removal
  - Exact duplicate detection via SHA-256 hashing
  - Document-level processing with error isolation
  - Financial notation preservation ($, %, numbers, etc.)
  - Deterministic processing with statistics tracking
  - Cleaning report generation (JSON)
  - Metadata preservation (JSONL)
  - Validation checks (boundaries, encoding, financial patterns)

- **Stage 3**: BPE implementation (pending)
- **Stage 4**: Benchmarking (pending)

## License

MIT License
