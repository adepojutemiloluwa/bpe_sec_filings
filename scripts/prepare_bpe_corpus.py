"""Script to run Stage 3: BPE corpus preparation."""

import argparse
import logging
import sys
from pathlib import Path

from sec_bpe.corpus.config import CorpusPreparationConfig
from sec_bpe.corpus.preparer import CorpusPreparer


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
    """Run Stage 3 corpus preparation.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Stage 3: BPE Corpus Preparation"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/cleaned",
        help="Directory containing Stage 2 cleaned output (default: data/cleaned)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/prepared",
        help="Directory for Stage 3 prepared output (default: data/prepared)",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="company",
        choices=["company", "document", "temporal"],
        help="Split strategy: company, document, or temporal (default: company)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Proportion for training (default: 0.80)",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
        help="Proportion for validation (default: 0.10)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Proportion for test (default: 0.10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--preserve-case",
        action="store_true",
        default=True,
        help="Preserve original case (default: True)",
    )
    parser.add_argument(
        "--no-preserve-case",
        action="store_false",
        dest="preserve_case",
        help="Do not preserve original case",
    )
    parser.add_argument(
        "--preserve-punctuation",
        action="store_true",
        default=True,
        help="Preserve punctuation (default: True)",
    )
    parser.add_argument(
        "--no-preserve-punctuation",
        action="store_false",
        dest="preserve_punctuation",
        help="Do not preserve punctuation",
    )
    parser.add_argument(
        "--boundary-mode",
        type=str,
        default="special",
        choices=["special", "remove"],
        help="Document boundary mode: special or remove (default: special)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--skip-statistics",
        action="store_true",
        help="Skip statistics generation (for faster runs)",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("SEC-BPE Stage 3: BPE Corpus Preparation")
    logger.info("=" * 60)

    try:
        # Create configuration
        config = CorpusPreparationConfig(
            split_strategy=args.split_strategy,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            preserve_case=args.preserve_case,
            preserve_punctuation=args.preserve_punctuation,
            boundary_mode=args.boundary_mode,
        )

        logger.info(f"Configuration:")
        logger.info(f"  Split strategy: {config.split_strategy}")
        logger.info(f"  Train ratio: {config.train_ratio}")
        logger.info(f"  Validation ratio: {config.validation_ratio}")
        logger.info(f"  Test ratio: {config.test_ratio}")
        logger.info(f"  Seed: {config.seed}")
        logger.info(f"  Preserve case: {config.preserve_case}")
        logger.info(f"  Preserve punctuation: {config.preserve_punctuation}")
        logger.info(f"  Boundary mode: {config.boundary_mode}")

        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")

        # Prepare corpus
        preparer = CorpusPreparer(args.input_dir, args.output_dir, config)
        report = preparer.prepare()

        logger.info("=" * 60)
        logger.info("Preparation Summary:")
        logger.info(f"  Documents loaded: {report['validation']['documents_loaded']}")
        logger.info(f"  Documents valid: {report['validation']['documents_valid']}")
        logger.info(f"  Documents skipped: {report['validation']['documents_skipped']}")
        logger.info(f"  Documents failed: {report['validation']['documents_failed']}")
        logger.info(f"  Total documents: {report['corpus_summary']['total_documents']}")
        logger.info(f"  Total companies: {report['corpus_summary']['total_companies']}")
        logger.info(f"  Total characters: {report['corpus_summary']['total_characters']:,}")
        logger.info(f"  Total bytes: {report['corpus_summary']['total_bytes']:,}")
        logger.info(f"  Unique characters: {report['corpus_summary']['unique_characters']}")
        logger.info("=" * 60)
        logger.info("Split Statistics:")
        for split_name, stats in report['splits'].items():
            if stats:
                logger.info(
                    f"  {split_name}: {stats['documents']} documents, "
                    f"{stats['companies']} companies, "
                    f"{stats['characters']:,} characters"
                )
        logger.info("=" * 60)

        # Check for leakage
        if report['leakage']['has_leakage']:
            logger.error("WARNING: Data leakage detected!")
            return 1

        # Check for errors
        if report['validation']['errors']:
            logger.error(f"WARNING: {len(report['validation']['errors'])} validation errors")
            for error in report['validation']['errors']:
                logger.error(f"  {error}")

        logger.info("Stage 3 completed successfully")
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
