"""Safe ZIP archive extraction for SEC filing ingestion."""

import hashlib
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from sec_bpe.corpus.ingestion_config import IngestionConfig


class ZipExtractor:
    """Extract ZIP archives safely for SEC filing ingestion."""

    def __init__(self, config: IngestionConfig):
        """Initialize the ZIP extractor.

        Args:
            config: Ingestion configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def extract_archives(self, source_dir: Path) -> Dict:
        """Extract all ZIP archives in the source directory.

        Args:
            source_dir: Directory containing ZIP archives.

        Returns:
            Extraction report dictionary.
        """
        report = {
            "enabled": self.config.extract_zip,
            "archives_discovered": 0,
            "archives_extracted": 0,
            "archives_skipped": 0,
            "archives_failed": 0,
            "files_extracted": 0,
            "errors": [],
            "extract_dir": str(source_dir / self.config.extract_dir),
        }

        if not self.config.extract_zip:
            self.logger.info("ZIP extraction disabled by configuration")
            return report

        # Find all ZIP files
        zip_files = sorted(source_dir.rglob("*.zip"))
        report["archives_discovered"] = len(zip_files)

        if not zip_files:
            self.logger.info("No ZIP archives found")
            return report

        # Create extraction directory
        extract_dir = source_dir / self.config.extract_dir
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Load extraction manifest to avoid re-extraction
        manifest = self._load_manifest(extract_dir)

        # Extract each archive
        for zip_path in zip_files:
            try:
                self._extract_single_archive(zip_path, extract_dir, manifest, report)
            except Exception as e:
                self.logger.error(f"Failed to extract {zip_path.name}: {e}")
                report["archives_failed"] += 1
                report["errors"].append(f"{zip_path.name}: {str(e)}")

        # Save updated manifest
        self._save_manifest(extract_dir, manifest)

        self.logger.info(
            f"ZIP extraction complete: {report['archives_extracted']} extracted, "
            f"{report['archives_skipped']} skipped, {report['archives_failed']} failed"
        )

        return report

    def _extract_single_archive(
        self,
        zip_path: Path,
        extract_dir: Path,
        manifest: Dict,
        report: Dict,
    ) -> None:
        """Extract a single ZIP archive.

        Args:
            zip_path: Path to ZIP archive.
            extract_dir: Directory to extract to.
            manifest: Extraction manifest for tracking.
            report: Report dictionary to update.
        """
        # Calculate archive hash
        archive_hash = self._calculate_hash(zip_path)

        # Check if already extracted
        if zip_path.name in manifest and manifest[zip_path.name] == archive_hash:
            self.logger.info(f"Skipping already extracted archive: {zip_path.name}")
            report["archives_skipped"] += 1
            return

        self.logger.info(f"Extracting archive: {zip_path.name}")

        # Open and validate archive
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Check for ZIP bomb
                self._check_zip_bomb(zf, zip_path)

                # Extract each file
                for member in zf.infolist():
                    if self._should_extract_member(member, zip_path):
                        self._extract_member(zf, member, extract_dir, zip_path, report)

        except zipfile.BadZipFile:
            raise ValueError(f"Corrupt ZIP archive: {zip_path.name}")

        # Update manifest
        manifest[zip_path.name] = archive_hash
        report["archives_extracted"] += 1

    def _should_extract_member(self, member: zipfile.ZipInfo, zip_path: Path) -> bool:
        """Check if a member should be extracted.

        Args:
            member: ZIP member info.
            zip_path: Path to ZIP archive (for error reporting).

        Returns:
            True if member should be extracted.
        """
        # Skip directories
        if member.is_dir():
            return False

        # Skip symbolic links and other special files
        if member.external_attr & 0xF0000000 == 0xA0000000:  # Symlink
            self.logger.warning(f"Skipping symbolic link in {zip_path.name}: {member.filename}")
            return False

        # Check file size limit
        if self.config.max_extracted_file_size > 0:
            if member.file_size > self.config.max_extracted_file_size:
                self.logger.warning(
                    f"Skipping oversized file in {zip_path.name}: "
                    f"{member.filename} ({member.file_size} bytes)"
                )
                return False

        return True

    def _extract_member(
        self,
        zf: zipfile.ZipFile,
        member: zipfile.ZipInfo,
        extract_dir: Path,
        zip_path: Path,
        report: Dict,
    ) -> None:
        """Extract a single member from the archive.

        Args:
            zf: ZIP file object.
            member: ZIP member info.
            extract_dir: Extraction directory.
            zip_path: Path to ZIP archive.
            report: Report dictionary to update.
        """
        # Validate path to prevent traversal attacks
        member_path = Path(member.filename)

        # Check for path traversal
        if ".." in member_path.parts or member_path.is_absolute():
            self.logger.warning(
                f"Skipping unsafe path in {zip_path.name}: {member.filename}"
            )
            return

        # Construct safe extraction path
        # Use archive name as subdirectory to avoid conflicts
        archive_subdir = extract_dir / zip_path.stem
        archive_subdir.mkdir(exist_ok=True)

        extract_path = archive_subdir / member_path.name

        # Handle duplicate filenames by adding suffix
        if extract_path.exists():
            base = extract_path.stem
            suffix = extract_path.suffix
            counter = 1
            while extract_path.exists():
                extract_path = archive_subdir / f"{base}_{counter}{suffix}"
                counter += 1

        # Extract the file
        try:
            with zf.open(member) as source, extract_path.open('wb') as target:
                target.write(source.read())

            report["files_extracted"] += 1
            self.logger.debug(f"Extracted: {member.filename} -> {extract_path.name}")

        except Exception as e:
            self.logger.error(f"Failed to extract {member.filename}: {e}")
            if extract_path.exists():
                extract_path.unlink()

    def _check_zip_bomb(self, zf: zipfile.ZipFile, zip_path: Path) -> None:
        """Check for ZIP bomb (excessive compression ratio).

        Args:
            zf: ZIP file object.
            zip_path: Path to ZIP archive.

        Raises:
            ValueError: If ZIP bomb detected.
        """
        if self.config.max_total_extracted_size == 0:
            return

        total_uncompressed = sum(m.file_size for m in zf.infolist())

        if total_uncompressed > self.config.max_total_extracted_size:
            raise ValueError(
                f"ZIP bomb detected in {zip_path.name}: "
                f"uncompressed size ({total_uncompressed} bytes) exceeds limit "
                f"({self.config.max_total_extracted_size} bytes)"
            )

    def _calculate_hash(self, path: Path) -> str:
        """Calculate SHA-256 hash of a file.

        Args:
            path: Path to file.

        Returns:
            Hexadecimal hash string.
        """
        sha256 = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _load_manifest(self, extract_dir: Path) -> Dict:
        """Load extraction manifest.

        Args:
            extract_dir: Extraction directory.

        Returns:
            Manifest dictionary.
        """
        manifest_path = extract_dir / ".extraction_manifest.json"
        if manifest_path.exists():
            import json
            try:
                return json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as e:
                self.logger.warning(f"Failed to load manifest: {e}")
        return {}

    def _save_manifest(self, extract_dir: Path, manifest: Dict) -> None:
        """Save extraction manifest.

        Args:
            extract_dir: Extraction directory.
            manifest: Manifest dictionary.
        """
        import json
        manifest_path = extract_dir / ".extraction_manifest.json"
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception as e:
            self.logger.warning(f"Failed to save manifest: {e}")
