"""Script to run Stage 2: SEC filing cleaning and normalization."""

import argparse
import logging
import sys
from pathlib import Path

from sec_bpe.preprocessing.corpus_builder import CleanedCorpusBuilder
from sec_bpe.preprocessing.validation import CorpusValidator


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
    """Run Stage 2 preprocessing pipeline.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Stage 2: SEC Filing Cleaning and Normalization"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing raw corpus from Stage 1 (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cleaned",
        help="Directory for cleaned corpus output (default: data/cleaned)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("SEC-BPE Stage 2: SEC Filing Cleaning and Normalization")
    logger.info("=" * 60)

    try:
        # Build cleaned corpus
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")

        builder = CleanedCorpusBuilder(args.input_dir, args.output_dir)
        stats = builder.build()

        logger.info("=" * 60)
        logger.info("Cleaning Statistics:")
        logger.info(f"  Documents processed: {stats['documents_processed']}")
        logger.info(f"  Documents cleaned: {stats['documents_cleaned']}")
        logger.info(f"  Documents failed: {stats['documents_failed']}")
        logger.info(f"  Duplicates detected: {stats['duplicates_detected']}")
        logger.info(f"  Total raw characters: {stats['total_raw_characters']:,}")
        logger.info(f"  Total cleaned characters: {stats['total_cleaned_characters']:,}")
        logger.info(f"  HTML elements removed: {stats['html_elements_removed']:,}")
        logger.info(f"  XBRL elements removed: {stats['xbrl_elements_removed']:,}")
        logger.info(f"  Tables processed: {stats['table_count']}")
        logger.info(f"  Encoding issues: {stats['encoding_issues']}")
        logger.info("=" * 60)

        # Validate corpus
        if not args.skip_validation:
            logger.info("Running validation checks...")
            validator = CorpusValidator()
            
            corpus_path = Path(args.output_dir) / "corpus.txt"
            metadata_path = Path(args.output_dir) / "metadata.jsonl"
            report_path = Path(args.output_dir) / "cleaning_report.json"

            validation_results = validator.validate_corpus(
                corpus_path, metadata_path, report_path
            )

            if validation_results["is_valid"]:
                logger.info("✓ Validation passed")
            else:
                logger.warning(f"⚠ Validation found {len(validation_results['errors'])} errors")
                for error in validation_results["errors"]:
                    logger.error(f"  - {error}")

            if validation_results["warnings"]:
                logger.warning(f"⚠ Validation found {len(validation_results['warnings'])} warnings")
                for warning in validation_results["warnings"]:
                    logger.warning(f"  - {warning}")

        logger.info("Stage 2 completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

