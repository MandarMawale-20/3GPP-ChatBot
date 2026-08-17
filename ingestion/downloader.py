"""Programmatic download of official 3GPP specification archives.

Only the allowlisted Rel-18 documents in `configs/corpus.yaml` may be
fetched, and only from the official 3GPP repository — never a mirror,
never a broader recursive crawl.

3GPP archive filenames follow `<spec_digits>-<release_letter><major><minor>.zip`,
e.g. `24501-i90.zip` for TS 24.501 version 18.9.0. The release letter maps
to a release number (Rel-18 = `i`); see `RELEASE_LETTERS`.

Version-digit encoding: the two characters after the release letter are
base-36 digits ('0'-'9' then 'a'-'z'), because a version component can
exceed 9. For example TS 23.501 archives run from `23501-i90.zip` (18.9.0)
through `23501-ia0.zip` (18.10.0), `23501-ib0.zip` (18.11.0),
`23501-ic0.zip` (18.12.0). Treating these as plain decimal digits fails to
recognize the newest archive once a version component passes 9.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# 3GPP release -> filename letter-code convention.
RELEASE_LETTERS: dict[int, str] = {
    15: "f",
    16: "g",
    17: "h",
    18: "i",
    19: "j",
    20: "k",
}

# Archive filename pattern, e.g. "24501-i90.zip", "23501-ic0.zip", or
# "23501-i50.docx" (some smaller specs are published as a bare .docx).
# Major/minor are single base-36 characters (digit or letter).
_ARCHIVE_NAME_RE = re.compile(
    r"^(?P<digits>\d{5})-(?P<letter>[a-z])(?P<major>[0-9a-z])(?P<minor>[0-9a-z])\.(?P<ext>zip|docx|doc)$",
    re.IGNORECASE,
)

_REQUEST_TIMEOUT_SECONDS = 30
_DOWNLOAD_CHUNK_BYTES = 1 << 16  # 64 KiB


class DownloadError(RuntimeError):
    """Raised when a document cannot be located or fetched safely."""


def base36_digit_to_int(char: str) -> int:
    """Convert a 3GPP version-digit character to its integer value:
    '0'-'9' -> 0-9, 'a'-'z' -> 10-35 (e.g. 'c' in `23501-ic0.zip` means
    major version 12).
    """
    char = char.lower()
    if char.isdigit():
        return int(char)
    if "a" <= char <= "z":
        return 10 + (ord(char) - ord("a"))
    raise ValueError(f"Invalid 3GPP version digit: {char!r}")


@dataclass(frozen=True)
class RemoteArchive:
    """A single candidate archive found in a directory listing."""

    filename: str
    url: str
    spec_digits: str  # e.g. "24501"
    release_letter: str  # e.g. "i"
    major: int  # decoded base-36 value, e.g. 'c' -> 12
    minor: int

    @property
    def version_sort_key(self) -> tuple[int, int]:
        return (self.major, self.minor)


def spec_number_to_digits(spec_number: str) -> str:
    """'24.501' -> '24501'. 3GPP filenames drop the dot."""
    return spec_number.replace(".", "")


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(requests.RequestException),
)
def fetch_directory_listing(series_url: str) -> str:
    """Fetch the raw HTML of a 3GPP series directory (e.g. .../24_series/).

    Retries on transient network errors only; a 404/permission error is
    not retried.
    """
    logger.info("Fetching directory listing: {}", series_url)
    response = requests.get(series_url, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def parse_archive_links(html: str, base_url: str) -> list[RemoteArchive]:
    """Extract archive links from a directory listing page by filename
    pattern only; no scraping of unrelated content.
    """
    soup = BeautifulSoup(html, "html.parser")
    archives: list[RemoteArchive] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        filename = href.rsplit("/", 1)[-1]
        match = _ARCHIVE_NAME_RE.match(filename)
        if not match:
            continue  # not a spec archive link (e.g. "../", index pages)

        archives.append(
            RemoteArchive(
                filename=filename,
                url=href if href.startswith("http") else f"{base_url.rstrip('/')}/{filename}",
                spec_digits=match.group("digits"),
                release_letter=match.group("letter").lower(),
                major=base36_digit_to_int(match.group("major")),
                minor=base36_digit_to_int(match.group("minor")),
            )
        )
    return archives


def find_latest_archive(
    archives: list[RemoteArchive], spec_number: str, release_number: int
) -> RemoteArchive:
    """Pick the highest-versioned archive for a spec within one release.

    Never falls back to a different release's letter code; mixing releases
    would defeat the release-isolation guarantee.
    """
    expected_digits = spec_number_to_digits(spec_number)
    expected_letter = RELEASE_LETTERS.get(release_number)
    if expected_letter is None:
        raise DownloadError(f"No known filename letter-code for release {release_number}")

    candidates = [
        a
        for a in archives
        if a.spec_digits == expected_digits and a.release_letter == expected_letter
    ]
    if not candidates:
        raise DownloadError(
            f"No archive found for spec {spec_number} release {release_number} "
            f"(letter '{expected_letter}') in directory listing"
        )

    return max(candidates, key=lambda a: a.version_sort_key)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(requests.RequestException),
)
def download_archive(archive: RemoteArchive, dest_path: Path) -> Path:
    """Stream-download an archive to disk.

    Writes to a `.part` file first and renames on success so a crash
    mid-download never leaves a corrupt file at the final path.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    logger.info("Downloading {} -> {}", archive.url, dest_path)
    with requests.get(archive.url, stream=True, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    f.write(chunk)

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise DownloadError(f"Downloaded file is empty: {archive.url}")

    tmp_path.rename(dest_path)
    logger.info("Downloaded {} ({} bytes)", dest_path.name, dest_path.stat().st_size)
    return dest_path
