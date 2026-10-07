"""Script to train BPE tokenizer."""

import argparse
import logging
import sys
from pathlib import Path

from sec_bpe.tokenizer.config import BPEConfig
from sec_bpe.tokenizer.reference_trainer import ReferenceBPETrainer
from sec_bpe.tokenizer.tokenizer import BPETokenizer
from sec_bpe.tokenizer.serializer import TokenizerSerializer
from sec_bpe.tokenizer.statistics import TrainingStatistics
from sec_bpe.tokenizer.validation import TokenizerValidator


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Whether to enable verbose logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    """Run BPE training.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Stage 4: BPE Tokenizer Training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to training corpus (train.txt from Stage 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for tokenizer artifacts",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Target vocabulary size (default: 8192)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency to learn merge (default: 2)",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="reference",
        choices=["reference", "optimized"],
        help="Trainer type (default: reference)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Stage 4: BPE Tokenizer Training")
    logger.info("=" * 60)

    try:
        # Validate input
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        output_path = Path(args.output)

        # Create configuration
        config = BPEConfig(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            trainer=args.trainer,
        )

        logger.info(f"Configuration:")
        logger.info(f"  Vocab size: {config.vocab_size}")
        logger.info(f"  Min frequency: {config.min_frequency}")
        logger.info(f"  Trainer: {config.trainer}")
        logger.info(f"  Special tokens: {config.special_tokens}")

        logger.info(f"Input corpus: {input_path}")
        logger.info(f"Output directory: {output_path}")

        # Initialize trainer
        trainer = ReferenceBPETrainer(config)

        # Load corpus
        trainer.load_corpus(input_path)

        # Record initial statistics
        stats = TrainingStatistics()
        corpus_bytes = input_path.stat().st_size
        corpus_chars = input_path.read_text(encoding="utf-8")
        stats.record_initial_state(
            vocab_size=trainer.vocabulary.size(),
            num_documents=len(trainer.documents),
            corpus_bytes=corpus_bytes,
            corpus_characters=len(corpus_chars),
            token_count=trainer.initial_token_count,
        )

        # Train
        logger.info("Starting BPE training...")
        vocabulary, merge_table = trainer.train()

        # Record final statistics
        stats.record_final_state(
            vocab_size=vocabulary.size(),
            token_count=trainer.final_token_count,
            min_frequency=config.min_frequency,
        )

        # Create tokenizer
        tokenizer = BPETokenizer(
            config=config,
            vocabulary=vocabulary,
            merge_table=merge_table,
        )

        # Save tokenizer
        logger.info("Saving tokenizer artifacts...")
        serializer = TokenizerSerializer()
        serializer.save(tokenizer, output_path)
        serializer.save_training_metadata(
            output_path,
            input_path,
            trainer.initial_token_count,
            trainer.final_token_count,
        )

        # Save statistics
        stats_path = output_path / "statistics.json"
        stats.save(stats_path)

        logger.info(f"Saved tokenizer to {output_path}")

        # Validate tokenizer
        logger.info("Validating tokenizer artifacts...")
        validator = TokenizerValidator(output_path)
        is_valid = validator.validate()

        if not is_valid:
            logger.error("Tokenizer validation failed!")
            for error in validator.errors:
                logger.error(f"  {error}")
            return 1

        logger.info("Tokenizer validation passed")

        # Print summary
        logger.info("=" * 60)
        logger.info("Training Summary:")
        logger.info(f"  Documents processed: {len(trainer.documents)}")
        logger.info(f"  Initial tokens: {trainer.initial_token_count:,}")
        logger.info(f"  Final tokens: {trainer.final_token_count:,}")
        logger.info(f"  Compression ratio: {trainer.initial_token_count / trainer.final_token_count:.2f}")
        logger.info(f"  Vocabulary size: {vocabulary.size()}")
        logger.info(f"  Merges learned: {merge_table.size()}")
        logger.info("=" * 60)

        # Test round-trip
        logger.info("Testing round-trip encoding/decoding...")
        test_text = "Revenue increased by 10%."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)

        if decoded == test_text:
            logger.info("Round-trip test passed")
        else:
            logger.error(f"Round-trip test failed: '{test_text}' != '{decoded}'")
            return 1

        logger.info("Stage 4 completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
