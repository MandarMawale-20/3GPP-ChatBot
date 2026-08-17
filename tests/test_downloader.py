from __future__ import annotations

import pytest

from ingestion.downloader import (
    DownloadError,
    find_latest_archive,
    parse_archive_links,
    spec_number_to_digits,
)

SAMPLE_DIRECTORY_HTML = """
<html><body>
<a href="24501-i80.zip">24501-i80.zip</a>
<a href="24501-i90.zip">24501-i90.zip</a>
<a href="24501-h80.zip">24501-h80.zip</a>
<a href="23501-i50.zip">23501-i50.zip</a>
<a href="../">Parent directory</a>
<a href="readme.txt">readme.txt</a>
</body></html>
"""


def test_spec_number_to_digits() -> None:
    assert spec_number_to_digits("24.501") == "24501"


def test_parse_archive_links_extracts_only_archive_files() -> None:
    archives = parse_archive_links(SAMPLE_DIRECTORY_HTML, "https://example.com/24_series/")
    filenames = {a.filename for a in archives}
    assert filenames == {"24501-i80.zip", "24501-i90.zip", "24501-h80.zip", "23501-i50.zip"}


def test_find_latest_archive_picks_highest_version_for_release() -> None:
    archives = parse_archive_links(SAMPLE_DIRECTORY_HTML, "https://example.com/24_series/")
    latest = find_latest_archive(archives, "24.501", release_number=18)
    assert latest.filename == "24501-i90.zip"
    assert latest.major == 9
    assert latest.minor == 0


def test_find_latest_archive_never_crosses_release_letters() -> None:
    # Rel-17's "h80" file must never be selected when Rel-18 ("i") is requested.
    archives = parse_archive_links(SAMPLE_DIRECTORY_HTML, "https://example.com/24_series/")
    latest = find_latest_archive(archives, "24.501", release_number=18)
    assert latest.release_letter == "i"


def test_find_latest_archive_raises_when_spec_absent() -> None:
    archives = parse_archive_links(SAMPLE_DIRECTORY_HTML, "https://example.com/24_series/")
    with pytest.raises(DownloadError):
        find_latest_archive(archives, "38.331", release_number=18)


def test_find_latest_archive_raises_for_unknown_release() -> None:
    archives = parse_archive_links(SAMPLE_DIRECTORY_HTML, "https://example.com/24_series/")
    with pytest.raises(DownloadError):
        find_latest_archive(archives, "24.501", release_number=99)
