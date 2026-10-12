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
from sec_bpe.corpus.ingestion_config import IngestionConfig

# Configure ingestion (optional)
config = IngestionConfig(
    extract_zip=True,  # Enable ZIP archive extraction
    max_extracted_file_size=100 * 1024 * 1024,  # 100 MB per file
    max_total_extracted_size=1024 * 1024 * 1024,  # 1 GB total
    extract_dir="_extracted",  # Extraction subdirectory
)

builder = CorpusBuilder(
    input_dir="data/raw",
    output_path="data/processed/sec_corpus.txt",
    config=config,  # Optional: defaults to extract_zip=False
)

document_count = builder.build()
print(f"Added {document_count} documents")
```

**ZIP Archive Extraction:**
- Set `extract_zip=True` to automatically extract ZIP archives before processing
- Extracted files are placed in `{input_dir}/_extracted/{archive_name}/`
- Original ZIP files are preserved
- Extraction is tracked via manifest to avoid re-extracting unchanged archives
- Safety features: path traversal protection, file size limits, ZIP bomb detection
- When `extract_zip=False` (default), ZIP files are skipped and only regular files are processed

### Stage 4: BPE Tokenizer Training

```bash
python scripts/train_bpe.py \
    --input data/prepared/train.txt \
    --output artifacts/tokenizer/sec_bpe_v1 \
    --vocab-size 8192 \
    --min-frequency 2
```

Or programmatically:

```python
from sec_bpe.tokenizer.config import BPEConfig
from sec_bpe.tokenizer.reference_trainer import ReferenceBPETrainer
from sec_bpe.tokenizer.tokenizer import BPETokenizer

config = BPEConfig(
    vocab_size=8192,
    min_frequency=2,
)

trainer = ReferenceBPETrainer(config)
trainer.load_corpus("data/prepared/train.txt")
vocabulary, merge_table = trainer.train()

tokenizer = BPETokenizer(config=config, vocabulary=vocabulary, merge_table=merge_table)
tokenizer.save("artifacts/tokenizer/sec_bpe_v1")

# Use the tokenizer
ids = tokenizer.encode("Revenue increased by 10%.")
text = tokenizer.decode(ids)
```

**Stage 4 Features:**
- Byte-level BPE implementation (no character-level assumptions)
- 256 base byte vocabulary + special tokens
- Deterministic pair selection with explicit tie-breaking
- Document boundary respect (no cross-document merges)
- Special token handling (`<DOCUMENT_START>`, `<DOCUMENT_END>`, `<UNK>`)
- Round-trip encoding/decoding (decode(encode(text)) == text)
- Comprehensive training statistics
- Artifact validation
- Merge inspection for learned patterns
- Configurable vocabulary size and minimum frequency

**Output Files:**
- `vocab.json` - Vocabulary with byte sequences
- `merges.json` - Ordered merge rules with frequencies
- `config.json` - Training configuration
- `tokenizer.json` - Tokenizer metadata
- `statistics.json` - Training statistics
- `training_metadata.json` - Corpus hash and training info

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

- **Stage 4**: BPE tokenizer training ✓
  - Byte-level BPE implementation from scratch
  - 256 base byte vocabulary
  - Special token system
  - Deterministic pair counting and selection
  - Explicit tie-breaking rules
  - Document boundary respect
  - Round-trip encoding/decoding
  - Training statistics and validation
  - Artifact serialization
  - Merge inspection

- **Stage 5**: Tokenizer evaluation (pending)
- **Stage 6**: Comparison against existing tokenizers (pending)

## License


MIT License
