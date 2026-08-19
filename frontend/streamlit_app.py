"""Streamlit frontend.

Run with:
    streamlit run frontend/streamlit_app.py

Talks to the FastAPI backend over HTTP and holds no retrieval/generation
logic itself, keeping the UI decoupled from the grounding pipeline.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

ALL_DOCUMENTS = "All Documents"
ALL_RELEASES = "All Releases"

st.set_page_config(page_title="3GPP Standards Assistant", layout="centered")

st.title("3GPP Standards Assistant")
st.caption("Grounded answers from the indexed 3GPP corpus")


@st.cache_data(ttl=300)
def fetch_documents() -> list[dict]:
    """Fetch the corpus allowlist from the backend (cached for 5 min)."""
    try:
        response = requests.get(f"{API_BASE_URL}/metadata/documents", timeout=10)
        response.raise_for_status()
        return response.json().get("documents", [])
    except requests.RequestException:
        return []


documents = fetch_documents()

col1, col2 = st.columns(2)
with col1:
    release = st.selectbox(
        "Release",
        options=[ALL_RELEASES, "Rel-18"],
        index=1,  # "Rel-18" is the default
        help="Search across all indexed releases, or restrict to a specific release.",
    )

# None means "all releases" (no release filter); a string like "Rel-18"
# scopes retrieval to that release only.
release_value = None if release == ALL_RELEASES else release

# Show only documents for the selected release, and de-duplicate entries
# that appear in more than one release (e.g. 24.501 in both Rel-17 and Rel-18).
if release_value is None:
    scoped_documents = documents
else:
    scoped_documents = [d for d in documents if d.get("release") == release_value]

seen_labels: set[str] = set()
doc_options = [ALL_DOCUMENTS]
for document in scoped_documents:
    label = f"TS {document['spec_number']} — {document['title']}"
    if label not in seen_labels:
        seen_labels.add(label)
        doc_options.append(label)

with col2:
    document_selection = st.selectbox(
        "Documents",
        options=doc_options,
        index=0,  # "All Documents" is the default
        help="Search across all indexed documents, or restrict to one specification.",
    )

st.caption(f"Scope: {release} · {document_selection}")

st.divider()

query = st.text_area(
    "Ask a question",
    placeholder="e.g. What does 23.501 say about AMF?",
    height=100,
)

asked = st.button("Ask", type="primary")

if asked and not query.strip():
    st.info("Enter a question to get an answer.")

if asked and query.strip():
    with st.spinner("Retrieving evidence and generating a grounded answer..."):
        try:
            request_body: dict = {"query": query, "release": release_value}
            # Only send spec_number when a specific document is selected —
            # omitting it (or sending null) searches across all documents
            # for the selected release in the backend.
            if document_selection != ALL_DOCUMENTS:
                request_body["spec_number"] = document_selection.split(" — ")[0].removeprefix("TS ").strip()
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json=request_body,
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
            result = None

    if result is not None:
        st.divider()
        if result["abstained"]:
            st.subheader("Answer")
            st.warning(result["answer"])
            if result.get("abstain_reason"):
                with st.expander("Why did the system abstain?"):
                    st.write(result["abstain_reason"])
        else:
            st.subheader("Answer")
            st.markdown(result["answer"])
            st.caption(f"Confidence: {result['confidence']:.3f}")

            if result["sources"]:
                st.divider()
                st.subheader("Sources")
                for source in result["sources"]:
                    st.markdown(
                        f"- **TS {source['spec_number']}** — Clause {source['clause']} "
                        f"({source['release']} v{source['version']})"
                    )

            elif not result["abstained"]:
                st.divider()
                st.caption("No sources were cited for this answer.")

elif not asked:
    st.caption("Ask a question above to receive a grounded answer with cited sources.")

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant answers from indexed 3GPP technical specifications "
        "across releases. It cites every claim and abstains "
        "rather than guessing when the indexed corpus doesn't have "
        "enough explicit evidence."
    )
    st.write(f"API: `{API_BASE_URL}`")