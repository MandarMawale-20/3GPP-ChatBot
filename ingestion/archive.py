"""Archive extraction.

Handles the ZIP -> DOC/DOCX step of the pipeline. Extraction is defensive
against zip-slip path traversal, since archives are fetched over the
network.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from loguru import logger


class ArchiveError(RuntimeError):
    """Raised for corrupt archives or archives with no usable document."""


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def safe_extract(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a ZIP archive, rejecting any entry that would escape dest_dir
    (defense-in-depth against zip-slip).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                member_path = dest_dir / member.filename
                if not _is_within_directory(dest_dir, member_path):
                    raise ArchiveError(f"Unsafe path in archive, refusing to extract: {member.filename}")
            archive.extractall(dest_dir)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"Corrupt ZIP archive: {zip_path}") from exc

    logger.info("Extracted {} -> {}", zip_path.name, dest_dir)
    return dest_dir


def find_document_file(directory: Path) -> Path:
    """Locate the DOC/DOCX specification file within an extracted archive.
    Archives typically contain one primary spec file plus occasional
    cover-sheet files, so the largest matching file is preferred.
    """
    candidates = sorted(directory.rglob("*.docx")) + sorted(directory.rglob("*.doc"))
    # Filter out common non-spec companions (e.g. macOS resource forks).
    candidates = [c for c in candidates if not c.name.startswith("~$") and not c.name.startswith("._")]

    if not candidates:
        raise ArchiveError(f"No DOC/DOCX file found in extracted archive: {directory}")

    chosen = max(candidates, key=lambda p: p.stat().st_size)
    logger.info("Selected document file: {}", chosen.name)
    return chosen
