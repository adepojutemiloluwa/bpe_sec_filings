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

```python
from sec_bpe.corpus.builder import CorpusBuilder

builder = CorpusBuilder(
    input_dir="data/raw",
    output_path="data/processed/sec_corpus.txt",
)

document_count = builder.build()
print(f"Added {document_count} documents")
```

## Development Status

- **Stage 1**: SEC filing corpus ingestion ✓
- **Stage 2**: SEC preprocessing and normalization (pending)
- **Stage 3**: BPE implementation (pending)
- **Stage 4**: Benchmarking (pending)

## License

MIT License
