"""DOC -> DOCX conversion.

Some older 3GPP archives contain legacy binary `.doc` files. `python-docx`
cannot read those directly, so we shell out to headless LibreOffice.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from loguru import logger


class ConversionError(RuntimeError):
    """Raised when LibreOffice is unavailable or the conversion fails."""


def is_libreoffice_available() -> bool:
    return shutil.which("soffice") is not None or shutil.which("libreoffice") is not None


def convert_doc_to_docx(doc_path: Path, output_dir: Path, timeout_seconds: int = 120) -> Path:
    """Convert a legacy .doc file to .docx using headless LibreOffice.

    Runs in a subprocess with an explicit timeout — LibreOffice can hang on
    malformed input, and this is called from an ingestion pipeline that
    should fail a single document rather than block indefinitely.
    """
    if doc_path.suffix.lower() != ".doc":
        raise ConversionError(f"Not a .doc file: {doc_path}")

    binary = shutil.which("soffice") or shutil.which("libreoffice")
    if binary is None:
        raise ConversionError(
            "LibreOffice not found on PATH. Install it (e.g. `apt install libreoffice`) "
            "to convert legacy .doc files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Converting DOC -> DOCX: {}", doc_path.name)

    try:
        result = subprocess.run(
            [
                binary,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(output_dir),
                str(doc_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConversionError(f"LibreOffice conversion timed out for {doc_path.name}") from exc

    converted_path = output_dir / (doc_path.stem + ".docx")
    if result.returncode != 0 or not converted_path.exists():
        raise ConversionError(
            f"LibreOffice conversion failed for {doc_path.name}: "
            f"returncode={result.returncode}, stderr={result.stderr[:500]}"
        )

    logger.info("Converted -> {}", converted_path.name)
    return converted_path
