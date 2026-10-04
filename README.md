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

### Stage 3: BPE Corpus Preparation

```bash
python scripts/prepare_bpe_corpus.py --input-dir data/cleaned --output-dir data/prepared
```

Or programmatically:

```python
from sec_bpe.corpus.config import CorpusPreparationConfig
from sec_bpe.corpus.preparer import CorpusPreparer

config = CorpusPreparationConfig(
    split_strategy="company",
    train_ratio=0.80,
    validation_ratio=0.10,
    test_ratio=0.10,
    seed=42,
)

preparer = CorpusPreparer("data/cleaned", "data/prepared", config)
report = preparer.prepare()
```

**Stage 3 Features:**
- Document-level loading and validation from Stage 2 output
- Multiple split strategies (company-aware, document-level, temporal)
- Leakage-free train/validation/test splits
- Deterministic splitting with configurable seed
- Comprehensive corpus statistics (characters, bytes, document lengths)
- Financial pattern analysis
- Reproducible split manifests
- Document boundary handling for BPE training
- Case and punctuation preservation (configurable)

**Output Files:**
- `train.txt`, `validation.txt`, `test.txt` - Split text corpora
- `train_metadata.jsonl`, `validation_metadata.jsonl`, `test_metadata.jsonl` - Per-split metadata
- `corpus_statistics.json` - Character, byte, and document statistics
- `split_manifest.json` - Reproducible split assignments
- `preparation_report.json` - Preparation summary and configuration

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

- **Stage 3**: BPE corpus preparation ✓
  - Document loading and validation from Stage 2 output
  - Multiple split strategies (company, document, temporal)
  - Leakage-free train/validation/test splits
  - Deterministic splitting with configurable seed
  - Comprehensive corpus statistics
  - Financial pattern analysis
  - Reproducible split manifests
  - Document boundary handling for BPE training
  - Case and punctuation preservation

- **Stage 4**: BPE implementation (pending)
- **Stage 5**: Benchmarking (pending)

## License


MIT License
