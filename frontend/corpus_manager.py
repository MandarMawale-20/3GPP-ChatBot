"""Corpus Management UI — non-technical discover / select / label / approve flow.

Run with:
    streamlit run frontend/corpus_manager.py

This is the *corpus-management* companion to `frontend/streamlit_app.py`
(the query UI). It lets a human browse the official 3GPP directories for an
*enabled* release, compare what exists against the approved allowlist in
`configs/corpus.yaml`, pick specs to approve via checkboxes, and enter the
official spec titles manually — then writes the approval through the same
comment-preserving, release-isolating writer the CLI uses
(`scripts.discover_corpus.add_to_corpus_yaml`).

Design constraints preserved from the rest of the system:
  - The allowlist choke point holds: only an *enabled* release can be browsed,
    and only specs actually present in the official 3GPP directory for that
    release can be approved (discovery already enforces release-isolation at the
    archive filename-letter level).
  - Titles are never invented: the title column is editable and must be filled
    in with the *official* 3GPP title before Approve. The UI will not approve a
    spec with a blank or placeholder title.
  - Approval is idempotent: re-approving an already-listed spec is a no-op.

The server-side data path reuses the CLI module directly (same as
`scripts/discover_corpus.py`), so the UI and CLI always agree on what
constitutes a valid approval.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Bootstrap so this script can import the project package tree regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from scripts.discover_corpus import (  # noqa: E402
    add_to_corpus_yaml,
    discover_specs,
    load_approved_specs_from,
)
from ingestion.downloader import RELEASE_LETTERS  # noqa: E402  (for the help note)

CORPUS_YAML = PROJECT_ROOT / "configs" / "corpus.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def _discovered(release: str, series: tuple[str, ...]) -> dict:
    """Cached discovery for one release. ttl bounds repeated network fetches."""
    return discover_specs(release=release, series=list(series) or None)


def _approved(release: str) -> set[str]:
    return set(load_approved_specs_from(CORPUS_YAML, release))


def _default_series() -> list[str]:
    """Series scan list shared with the CLI so the UI and CLI agree."""
    import scripts.discover_corpus as dc

    return list(dc.DEFAULT_SERIES)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="3GPP Corpus Manager",
    layout="wide",
)

st.title("3GPP Corpus Manager")
st.caption(
    "Browse the official 3GPP directory for a release, compare it against the "
    "approved allowlist, and approve verified specs by their **official** title."
)

config = get_settings()
enabled_releases = list(config.enabled_releases)
default_release = config.default_release

# ---- Release selector -----------------------------------------------------
if not enabled_releases:
    st.error(
        "No releases are enabled in `configs/corpus.yaml`. Enable one (and "
        "populate its allowlist) before managing the corpus."
    )
    st.stop()

release = st.segmented_control(
    "Release to manage",
    options=enabled_releases,
    default=default_release,
    help=(
        "Only enabled releases can be browsed or edited. Release isolation is "
        "enforced at the archive filename-letter level "
        f"(REL_LETTERS: {', '.join(f'{k}={v}' for k, v in sorted(RELEASE_LETTERS.items()))}), "
        "so a release's archives never mix with another release's."
    ),
)

if release is None:
    st.stop()

# ---- Series override (advanced) ------------------------------------------
series_override = st.text_input(
    "Series directories to scan (optional override)",
    value="",
    help=(
        "Leave blank to scan the default series "
        f"({', '.join(_default_series())}). Comma-separated to add more, e.g. `22,26`."
    ),
)
series = (
    tuple(s.strip() for s in series_override.split(",") if s.strip())
    if series_override.strip()
    else tuple(_default_series())
)

# ---- Discover --------------------------------------------------------------
if st.button("Discover official specs", type="primary", use_container_width=True):
    st.session_state["discovered"] = True

discovered = None
if st.session_state.get("discovered"):
    with st.spinner(f"Fetching official 3GPP directory for {release}..."):
        discovered = _discovered(release, series)
    if not discovered:
        st.warning(f"No specifications found for {release} in the scanned series.")
    else:
        st.success(f"Found **{len(discovered)}** specifications for {release}.")

# ---- Browse-all-releases summary (read-only) ------------------------------
if st.button("Show discovered counts across all enabled releases"):
    with st.spinner("Browsing all enabled releases..."):
        counts = {
            rel: len(discover_specs(rel, list(series) or None))
            for rel in enabled_releases
        }
    st.dataframe(
        pd.DataFrame([{"release": r, "discovered": n} for r, n in counts.items()]),
        use_container_width=True,
    )

# ---- Approval grid --------------------------------------------------------
if discovered:
    approved = _approved(release)
    rows = []
    for spec in sorted(discovered.values(), key=lambda s: s.spec_number):
        is_approved = spec.spec_number in approved
        rows.append(
            {
                "select": (not is_approved),
                "spec_number": spec.spec_number,
                "series": spec.series,
                "version": spec.version,
                "title": "",  # user fills this in — official title only
                "status": "approved" if is_approved else "new / not approved",
            }
        )
    df = pd.DataFrame(rows)

    st.markdown(
        "Mark the specs you want to approve, then **enter each spec's official "
        "3GPP title** in the `title` column. Specs already approved are shown "
        "as `approved` and are pre-unselected. Titles are never invented — a "
        "spec with a blank title will not be approved."
    )

    edited = st.data_editor(
        df,
        key="editor",
        use_container_width=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Select", default=False),
            "title": st.column_config.TextColumn(
                "Official title (required)",
                placeholder="e.g. Non-Access-Stratum (NAS) protocol for 5G System (5GS)",
                help=(
                    "Paste the official 3GPP title verbatim from the spec. "
                    "Titles are never invented — a spec with a blank title "
                    "will not be approved."
                ),
            ),
            "spec_number": st.column_config.TextColumn("Spec number", disabled=True),
            "series": st.column_config.TextColumn("Series", disabled=True),
            "version": st.column_config.TextColumn("Version", disabled=True),
            "status": st.column_config.TextColumn("Status", disabled=True),
        },
        hide_index=True,
    )

    to_approve = [
        r
        for _, r in edited.iterrows()
        if r["select"] and r["status"] != "approved"
    ]

    if st.button(
        f"Approve {len(to_approve)} selected spec(s) into {release}",
        type="primary",
        use_container_width=True,
        disabled=not to_approve,
    ):
        additions: list[tuple[str, str, str]] = []
        missing_titles: list[str] = []
        for r in to_approve:
            spec = r["spec_number"]
            title = (r["title"] or "").strip()
            if not title:
                missing_titles.append(spec)
                continue
            if spec not in discovered:
                st.error(f"Specification {spec} is not in the discovered set for {release}.")
                continue
            additions.append((spec, title, discovered[spec].series))

        if missing_titles:
            st.error(
                "Refusing to approve specs with blank titles: "
                + ", ".join(missing_titles)
            )
        else:
            # Invalidate the discovery cache so the allowlist reflects new state.
            _discovered.clear()
            add_to_corpus_yaml(additions, release, path=CORPUS_YAML)
            st.session_state.pop("discovered", None)
            st.session_state.pop("editor", None)
            st.success(
                f"Approved **{len(additions)}** specification(s) into `{release}` "
                "in `configs/corpus.yaml`."
            )
            st.markdown(
                "Review the change with `git diff configs/corpus.yaml`, then run "
                f"the ingestion pipeline, e.g. `python scripts/download.py --release "
                f"{release} --missing` followed by `preprocess` / `ingest_qdrant`."
            )

else:
    st.info("Click **Discover official specs** to load the comparison grid.")
