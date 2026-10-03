"""Split documents into train/validation/test sets."""

import logging
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set

from sec_bpe.corpus.config import CorpusPreparationConfig


class CorpusSplitter:
    """Split documents into train/validation/test sets."""

    def __init__(self, config: CorpusPreparationConfig):
        """Initialize the corpus splitter.

        Args:
            config: Corpus preparation configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        random.seed(config.seed)

    def split(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Split documents into train/validation/test sets.

        Args:
            documents: List of document dictionaries.

        Returns:
            Dictionary with keys "train", "validation", "test" containing
            lists of document dictionaries.
        """
        if self.config.split_strategy == "company":
            return self._split_by_company(documents)
        elif self.config.split_strategy == "document":
            return self._split_by_document(documents)
        elif self.config.split_strategy == "temporal":
            return self._split_by_temporal(documents)
        else:
            raise ValueError(f"Unknown split strategy: {self.config.split_strategy}")

    def _split_by_company(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Split documents by company/ticker.

        Groups all documents from the same company together.
        Then assigns companies to splits.

        Args:
            documents: List of document dictionaries.

        Returns:
            Dictionary with train/validation/test document lists.
        """
        # Group documents by ticker
        company_docs = defaultdict(list)
        for doc in documents:
            ticker = doc["metadata"].get("ticker", "UNKNOWN")
            company_docs[ticker].append(doc)

        # Get list of companies
        companies = list(company_docs.keys())
        random.shuffle(companies)

        # Calculate split points
        n_companies = len(companies)
        n_train = int(n_companies * self.config.train_ratio)
        n_val = int(n_companies * self.config.validation_ratio)

        # Assign companies to splits
        train_companies = set(companies[:n_train])
        val_companies = set(companies[n_train:n_train + n_val])
        test_companies = set(companies[n_train + n_val:])

        # Assign documents to splits
        splits = {"train": [], "validation": [], "test": []}
        company_assignments = {}

        for ticker, docs in company_docs.items():
            if ticker in train_companies:
                splits["train"].extend(docs)
                company_assignments[ticker] = "train"
            elif ticker in val_companies:
                splits["validation"].extend(docs)
                company_assignments[ticker] = "validation"
            elif ticker in test_companies:
                splits["test"].extend(docs)
                company_assignments[ticker] = "test"

        self.logger.info(
            f"Company split: {len(train_companies)} train, "
            f"{len(val_companies)} validation, {len(test_companies)} test"
        )

        # Store company assignments for manifest
        self.company_assignments = company_assignments

        return splits

    def _split_by_document(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Split documents randomly at the document level.

        Each document is independently assigned to a split.

        Args:
            documents: List of document dictionaries.

        Returns:
            Dictionary with train/validation/test document lists.
        """
        # Shuffle documents
        shuffled = documents.copy()
        random.shuffle(shuffled)

        # Calculate split points
        n_docs = len(shuffled)
        n_train = int(n_docs * self.config.train_ratio)
        n_val = int(n_docs * self.config.validation_ratio)

        # Assign documents to splits
        splits = {
            "train": shuffled[:n_train],
            "validation": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:],
        }

        self.logger.info(
            f"Document split: {len(splits['train'])} train, "
            f"{len(splits['validation'])} validation, {len(splits['test'])} test"
        )

        return splits

    def _split_by_temporal(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Split documents by filing date.

        Older filings go to train, middle to validation, newest to test.

        Args:
            documents: List of document dictionaries.

        Returns:
            Dictionary with train/validation/test document lists.
        """
        # Filter documents with valid filing dates
        dated_docs = []
        for doc in documents:
            filing_date = doc["metadata"].get("filing_date")
            if filing_date:
                try:
                    date = datetime.strptime(filing_date, "%Y-%m-%d")
                    dated_docs.append((date, doc))
                except ValueError:
                    self.logger.warning(
                        f"Invalid filing date format for {doc['document_id']}: {filing_date}"
                    )

        if not dated_docs:
            self.logger.warning("No documents with valid filing dates, falling back to document split")
            return self._split_by_document(documents)

        # Sort by date
        dated_docs.sort(key=lambda x: x[0])

        # Extract sorted documents
        sorted_docs = [doc for _, doc in dated_docs]

        # Calculate split points
        n_docs = len(sorted_docs)
        n_train = int(n_docs * self.config.train_ratio)
        n_val = int(n_docs * self.config.validation_ratio)

        # Assign documents to splits
        splits = {
            "train": sorted_docs[:n_train],
            "validation": sorted_docs[n_train:n_train + n_val],
            "test": sorted_docs[n_train + n_val:],
        }

        # Log date ranges
        if splits["train"]:
            train_start = dated_docs[0][0].strftime("%Y-%m-%d")
            train_end = dated_docs[n_train - 1][0].strftime("%Y-%m-%d")
            self.logger.info(f"Train date range: {train_start} to {train_end}")

        if splits["validation"]:
            val_start = dated_docs[n_train][0].strftime("%Y-%m-%d")
            val_end = dated_docs[n_train + n_val - 1][0].strftime("%Y-%m-%d")
            self.logger.info(f"Validation date range: {val_start} to {val_end}")

        if splits["test"]:
            test_start = dated_docs[n_train + n_val][0].strftime("%Y-%m-%d")
            test_end = dated_docs[-1][0].strftime("%Y-%m-%d")
            self.logger.info(f"Test date range: {test_start} to {test_end}")

        return splits

    def check_leakage(self, splits: Dict[str, List[Dict]]) -> Dict:
        """Check for data leakage between splits.

        Args:
            splits: Dictionary with train/validation/test document lists.

        Returns:
            Leakage report dictionary.
        """
        report = {
            "document_leakage": [],
            "company_leakage": [],
            "has_leakage": False,
        }

        # Get document IDs per split
        doc_ids = {
            "train": {doc["document_id"] for doc in splits["train"]},
            "validation": {doc["document_id"] for doc in splits["validation"]},
            "test": {doc["document_id"] for doc in splits["test"]},
        }

        # Check document leakage
        train_val_overlap = doc_ids["train"] & doc_ids["validation"]
        if train_val_overlap:
            report["document_leakage"].append(
                f"train/validation overlap: {len(train_val_overlap)} documents"
            )
            report["has_leakage"] = True

        train_test_overlap = doc_ids["train"] & doc_ids["test"]
        if train_test_overlap:
            report["document_leakage"].append(
                f"train/test overlap: {len(train_test_overlap)} documents"
            )
            report["has_leakage"] = True

        val_test_overlap = doc_ids["validation"] & doc_ids["test"]
        if val_test_overlap:
            report["document_leakage"].append(
                f"validation/test overlap: {len(val_test_overlap)} documents"
            )
            report["has_leakage"] = True

        # Check company leakage (if company split was used)
        if hasattr(self, "company_assignments"):
            companies = {
                "train": set(),
                "validation": set(),
                "test": set(),
            }

            for split_name in ["train", "validation", "test"]:
                for doc in splits[split_name]:
                    ticker = doc["metadata"].get("ticker", "UNKNOWN")
                    companies[split_name].add(ticker)

            train_val_company_overlap = companies["train"] & companies["validation"]
            if train_val_company_overlap:
                report["company_leakage"].append(
                    f"train/validation company overlap: {len(train_val_company_overlap)} companies"
                )
                report["has_leakage"] = True

            train_test_company_overlap = companies["train"] & companies["test"]
            if train_test_company_overlap:
                report["company_leakage"].append(
                    f"train/test company overlap: {len(train_test_company_overlap)} companies"
                )
                report["has_leakage"] = True

            val_test_company_overlap = companies["validation"] & companies["test"]
            if val_test_company_overlap:
                report["company_leakage"].append(
                    f"validation/test company overlap: {len(val_test_company_overlap)} companies"
                )
                report["has_leakage"] = True

        return report
