"""Document version validation.

Filename alone must never be trusted as the version — every downloaded
file is checked against the expected spec/release before it is allowed
into the raw store, and validation failure means "do not index", not
"index with a warning".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ingestion.downloader import RELEASE_LETTERS, base36_digit_to_int, spec_number_to_digits


class VersionValidationError(RuntimeError):
    """Raised when a downloaded document does not match the expected
    specification / release / version. Callers must treat this as a hard
    stop: do not index.
    """


@dataclass(frozen=True)
class ExpectedDocument:
    spec_number: str
    release: str  # e.g. "Rel-18"
    release_number: int  # e.g. 18


@dataclass(frozen=True)
class ValidatedDocument:
    spec_number: str
    release: str
    release_number: int
    version: str  # e.g. "18.9.0"
    filename: str
    sha256: str


def compute_sha256(path: Path) -> str:
    """Stream-hash a file rather than reading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_version_from_filename(filename: str) -> tuple[str, int, int]:
    """'24501-i90.docx' -> ('i', 9, 0); '23501-ic0.zip' -> ('i', 12, 0).

    The two characters after the release letter are base-36 digits (see
    the version-digit encoding note in ingestion/downloader.py). Kept
    separate from the archive-discovery regex so out-of-band files can be
    validated through the same code path.
    """
    stem = Path(filename).stem  # strips .zip/.docx/.doc
    parts = stem.split("-")
    if len(parts) != 2 or len(parts[1]) != 3:
        raise VersionValidationError(f"Filename does not match 3GPP convention: {filename!r}")

    letter = parts[1][0]
    try:
        major = base36_digit_to_int(parts[1][1])
        minor = base36_digit_to_int(parts[1][2])
    except ValueError as exc:
        raise VersionValidationError(f"Invalid version digits in: {filename!r}") from exc

    return letter, major, minor


def validate_document(
    filename: str,
    file_path: Path,
    expected: ExpectedDocument,
) -> ValidatedDocument:
    """Validate a downloaded file against the expected spec/release.

    Single choke point every raw file must pass before being trusted. On
    any mismatch we raise rather than log-and-continue.
    """
    expected_digits = spec_number_to_digits(expected.spec_number)
    actual_digits = Path(filename).stem.split("-")[0]

    if actual_digits != expected_digits:
        raise VersionValidationError(
            f"Spec number mismatch: expected {expected_digits}, filename has {actual_digits}"
        )

    letter, major, minor = parse_version_from_filename(filename)
    expected_letter = RELEASE_LETTERS.get(expected.release_number)
    if letter != expected_letter:
        raise VersionValidationError(
            f"Release mismatch for {filename!r}: expected letter "
            f"{expected_letter!r} (release {expected.release_number}), got {letter!r}"
        )

    if not file_path.exists() or file_path.stat().st_size == 0:
        raise VersionValidationError(f"File missing or empty: {file_path}")

    sha256 = compute_sha256(file_path)
    version = f"{expected.release_number}.{major}.{minor}"

    logger.info(
        "Version validated: {} / {} (sha256={}...)",
        expected.release,
        version,
        sha256[:12],
    )

    return ValidatedDocument(
        spec_number=expected.spec_number,
        release=expected.release,
        release_number=expected.release_number,
        version=version,
        filename=filename,
        sha256=sha256,
    )
