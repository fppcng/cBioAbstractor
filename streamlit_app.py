"""
cBioAbstractor — Streamlit Application
Self-contained Streamlit app for cBioPortal curation support.

This cleaned version keeps only:
  1. Curation Report
  2. File Classification

It removes merge-conflict markers, Docker/backend assumptions, and api_config.py usage.
Set ANTHROPIC_API_KEY as an environment variable or Streamlit secret.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="cBioAbstractor",
    page_icon="🧬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# API key loading
# Resolution order:
#   1. ANTHROPIC_API_KEY environment variable
#   2. Streamlit secrets
#   3. Sidebar input
# ─────────────────────────────────────────────────────────────────────────────
def _load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key

    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            return key
    except Exception:
        pass

    return ""


_API_KEY = _load_api_key()


def _get_api_key() -> str:
    return (_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")).strip()


def _require_api_key() -> bool:
    if not _get_api_key():
        st.error("Please add your Anthropic API key in the sidebar or set ANTHROPIC_API_KEY.")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save_upload_to_tmp(uploaded_file, filename: str | None = None) -> str:
    tmp_dir = tempfile.mkdtemp()
    safe_name = filename or uploaded_file.name
    path = os.path.join(tmp_dir, safe_name)
    with open(path, "wb") as handle:
        handle.write(uploaded_file.getvalue())
    return path


def _safe_cleanup(*paths: str) -> None:
    for path in paths:
        if not path:
            continue
        try:
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        except Exception:
            pass


def _call_anthropic_with_retry(
    client,
    model: str,
    system: str,
    user_content: str,
    max_tokens: int = 2000,
    retries: int = 3,
    backoff: float = 5.0,
) -> str:
    import anthropic

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            return response.content[0].text
        except anthropic.RateLimitError as exc:
            last_error = exc
            time.sleep(backoff * (attempt + 1))
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                last_error = exc
                time.sleep(backoff * (attempt + 1))
            else:
                raise
        except anthropic.APIConnectionError as exc:
            last_error = exc
            time.sleep(backoff * (attempt + 1))

    raise last_error or RuntimeError("Anthropic API call failed after retries.")


def _parse_llm_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[^\n]*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def _looks_tmp(name: str) -> bool:
    return bool(re.match(r"^tmp[a-z0-9_]{4,}", os.path.splitext(name)[0], re.I))


def _colour_curability(value: str) -> str:
    return {
        "Yes": "background-color:#E2EFDA;color:#375623",
        "Partly curatable": "background-color:#FFF2CC;color:#7F6000",
        "Needs manual intervention": "background-color:#FCE4D6;color:#843C0C",
    }.get(value, "")


def _colour_priority(value: str) -> str:
    return {
        "HIGH": "background-color:#FCE4D6;color:#843C0C",
        "MEDIUM": "background-color:#FFF2CC;color:#7F6000",
        "LOW": "background-color:#E2EFDA;color:#375623",
        "N/A": "background-color:#F2F2F2;color:#595959",
    }.get(value, "")


def _colour_confidence(value: str) -> str:
    try:
        numeric = float(str(value).replace("%", ""))
        if numeric >= 70:
            return "background-color:#E2EFDA;color:#375623"
        if numeric >= 40:
            return "background-color:#FFF2CC;color:#7F6000"
        return "background-color:#FCE4D6;color:#843C0C"
    except Exception:
        return ""


def _curability_label(value: str) -> str:
    return {
        "YES": "Yes",
        "PARTIAL": "Partly curatable",
        "NO": "Needs manual intervention",
    }.get(value, value or "—")


def _format_label(value: str) -> str:
    return {
        "NOT_LOADABLE": "Needs manual intervention",
        "Not directly loadable": "Needs manual intervention",
    }.get(value, value or "—")


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────
def _render_inline_report(meta: dict[str, Any], summary: dict[str, Any]) -> None:
    st.markdown("## Study Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Study ID", summary.get("study_id") or "—")
    col2.metric("Cancer Type", summary.get("cancer_type") or "—")
    col3.metric("Samples", summary.get("num_samples") or "—")
    col4.metric("Reference Genome", summary.get("reference_genome") or "—")

    fields = [
        ("Title", meta.get("study_title")),
        ("Cancer type full", meta.get("cancer_type_full")),
        ("Primary site", meta.get("primary_site")),
        ("Publication", " ".join(str(x) for x in [meta.get("journal", ""), meta.get("year", "")] if x).strip()),
        ("PMID", meta.get("pmid")),
        ("DOI", meta.get("doi")),
        ("First author", meta.get("first_author_surname")),
        ("Corresponding author(s)", meta.get("corresponding_authors")),
        ("Cohort", meta.get("cohort_description")),
        ("Summary", meta.get("description")),
    ]
    for label, value in fields:
        if value:
            st.markdown(f"**{label}:** {value}")

    sequencing_types = meta.get("sequencing_types")
    if sequencing_types:
        if isinstance(sequencing_types, list):
            sequencing_types = ", ".join(str(x) for x in sequencing_types)
        st.markdown(f"**Sequencing:** {sequencing_types}")

    repositories = meta.get("data_repositories")
    if repositories:
        if isinstance(repositories, list):
            repositories = ", ".join(str(x) for x in repositories)
        st.markdown(f"**Data repositories:** {repositories}")

    key_findings = meta.get("key_findings") or []
    if key_findings:
        st.markdown("**Key findings:**")
        for finding in key_findings:
            st.markdown(f"- {finding}")

    st.divider()
    st.markdown("## Supplementary File Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("High Priority", summary.get("high_priority", 0))
    col2.metric("Medium Priority", summary.get("medium_priority", 0))
    col3.metric("Needs Manual Intervention", summary.get("not_loadable", 0))

    breakdown = summary.get("file_breakdown", []) or []
    if not breakdown:
        st.info("No supplementary file breakdown was generated.")
        return

    table = pd.DataFrame([
        {
            "File": row.get("file", "—"),
            "Sheet": row.get("sheet", "—"),
            "cBioPortal Format": _format_label(row.get("cbio_format", "—")),
            "Confidence": f"{float(row.get('confidence', 0)):.0f}%",
            "Loadable": _curability_label(row.get("curability", "—")),
            "Priority": row.get("priority", "—"),
            "Columns Present": ", ".join(row.get("req_present", [])) or "—",
            "Columns Missing": ", ".join(row.get("req_missing", [])) or "None",
        }
        for row in breakdown
    ])

    styled = (
        table.style
        .map(_colour_curability, subset=["Loadable"])
        .map(_colour_priority, subset=["Priority"])
        .map(_colour_confidence, subset=["Confidence"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("## Per-Sheet Classification Detail")
    for row in breakdown:
        label = f"{row.get('file', '—')} — {row.get('sheet', '—')}"
        with st.expander(label, expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Format:** {_format_label(row.get('cbio_format', '—'))}")
            c2.markdown(f"**Confidence:** {float(row.get('confidence', 0)):.0f}%")
            c3.markdown(f"**Priority:** {row.get('priority', '—')}")

            if row.get("verdict"):
                st.markdown(f"**Assessment:** {row['verdict']}")
            if row.get("req_present"):
                st.success("Required columns found: " + ", ".join(row["req_present"]))
            if row.get("req_missing"):
                st.warning("Required columns missing: " + ", ".join(row["req_missing"]))
            if row.get("opt_present"):
                st.info("Optional columns found: " + ", ".join(row["opt_present"]))

    st.divider()
    st.markdown("## Suggested Study Metadata")
    meta_rows = {
        "cancer_study_identifier": summary.get("study_id") or "—",
        "name": meta.get("study_title") or "—",
        "description": meta.get("meta_description") or meta.get("description") or "—",
        "cancer_type": meta.get("cancer_type") or "—",
        "short_name": meta.get("study_id_suggestion") or "—",
        "pmid": meta.get("pmid") or "—",
        "groups": "PUBLIC",
    }
    meta_df = pd.DataFrame([{"Field": key, "Value": value} for key, value in meta_rows.items()])
    st.dataframe(meta_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("cBioAbstractor")
    st.caption("Automated curation support for cancer genomics studies.")
    st.divider()

    entered_key = st.text_input(
        "Anthropic API key",
        type="password",
        value="" if _get_api_key() else "",
        placeholder="sk-ant-...",
        help="For local use, you can also set ANTHROPIC_API_KEY in your shell.",
    )
    if entered_key:
        os.environ["ANTHROPIC_API_KEY"] = entered_key.strip()
        _API_KEY = entered_key.strip()

    if _get_api_key():
        st.success("Connected")
    else:
        st.warning("API key not configured")

    st.divider()
    st.caption("Version 1.2 — Streamlit only")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("cBioAbstractor")
st.markdown(
    "Upload a published cancer genomics paper and supplementary data files "
    "to generate a structured cBioPortal curation summary."
)


tab_curate, tab_detect = st.tabs(["Curation Report", "File Classification"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Curation Report
# ═════════════════════════════════════════════════════════════════════════════
with tab_curate:
    st.subheader("Curation Report Generator")
    st.markdown(
        "Upload the main paper PDF and supplementary files. The tool extracts "
        "study metadata and classifies each file against cBioPortal formats."
    )

    try:
        from spec_fetcher import fetch_spec

        spec_info = fetch_spec()
        if spec_info.get("source") == "live":
            st.caption(f"Format specifications loaded from live cBioPortal docs ({len(spec_info.get('specs', []))} formats).")
        else:
            st.caption("Using embedded cBioPortal format specifications.")
    except Exception:
        st.caption("Using embedded cBioPortal format specifications.")

    st.divider()
    col_pdf, col_supp = st.columns(2)
    with col_pdf:
        paper_pdf = st.file_uploader("Main paper PDF", type=["pdf"], key="paper_pdf")
    with col_supp:
        supp_files = st.file_uploader(
            "Supplementary files",
            type=["xlsx", "xls", "csv", "tsv", "txt", "tab", "maf", "doc", "docx", "pdf"],
            accept_multiple_files=True,
            key="supp_files",
        )

    if supp_files:
        st.markdown("#### Confirm uploaded filenames")
        st.caption("Edit names here only if Streamlit shows temporary filenames.")
        cols = st.columns(min(len(supp_files), 3))
        for idx, file in enumerate(supp_files):
            ext = os.path.splitext(file.name)[1] or ".xlsx"
            auto_name = f"Supplementary_Data_{idx + 1}{ext}"
            default = auto_name if _looks_tmp(file.name) else file.name
            st.text_input(f"File {idx + 1}", value=default, key=f"fname_{idx}")

    with st.expander("Options"):
        model = st.selectbox(
            "Anthropic model",
            options=[
                "claude-sonnet-4-20250514",
                "claude-3-5-haiku-20241022",
                "claude-3-5-sonnet-20241022",
                "claude-sonnet-4-6",            # latest Sonnet — recommended default
                "claude-opus-4-6",              # most capable, slower/more expensive
                "claude-haiku-4-5-20251001",    # fastest/cheapest
            ],
            index=0,
        )

    if st.button("Generate Curation Report", disabled=paper_pdf is None, type="primary"):
        if not _require_api_key():
            st.stop()

        pdf_tmp: str | None = None
        supp_tmps: list[str] = []

        try:
            with st.spinner("Saving uploaded files..."):
                pdf_tmp = _save_upload_to_tmp(paper_pdf)
                for idx, uploaded in enumerate(supp_files or []):
                    filename = st.session_state.get(f"fname_{idx}") or uploaded.name
                    supp_tmps.append(_save_upload_to_tmp(uploaded, filename=filename))

            with st.spinner("Step 1 of 2 — Extracting study metadata from PDF..."):
                import anthropic
                from cbioportal_curator import SYSTEM_PROMPT_CURATOR, _extract_pdf_text

                pdf_text = _extract_pdf_text(pdf_tmp)
                meta: dict[str, Any] = {}
                if pdf_text.strip():
                    client = anthropic.Anthropic(api_key=_get_api_key())
                    raw_meta = _call_anthropic_with_retry(
                        client=client,
                        model=model,
                        system=SYSTEM_PROMPT_CURATOR,
                        user_content=pdf_text[:40000],
                        max_tokens=2000,
                    )
                    try:
                        meta = _parse_llm_json(raw_meta)
                    except Exception:
                        st.warning("Metadata extraction returned unexpected format. Continuing with file classification.")
                        meta = {}
                else:
                    st.warning("Could not extract text from the PDF. Metadata fields will be blank.")

            with st.spinner(f"Step 2 of 2 — Classifying {len(supp_tmps)} supplementary file(s)..."):
                from cbioportal_curator import _analyse_supplementary_files

                records = _analyse_supplementary_files(supp_tmps)

            summary = {
                "study_id": meta.get("study_id_suggestion") or "—",
                "cancer_type": meta.get("cancer_type") or "—",
                "num_samples": meta.get("num_samples") or "—",
                "reference_genome": meta.get("reference_genome") or "—",
                "files_analysed": len(supp_tmps),
                "sheets_analysed": len(records),
                "high_priority": sum(1 for r in records if r.get("priority") == "HIGH"),
                "medium_priority": sum(1 for r in records if r.get("priority") == "MEDIUM"),
                "not_loadable": sum(1 for r in records if r.get("curability") == "NO"),
                "file_breakdown": [
                    {
                        "file": r.get("file", "—"),
                        "sheet": r.get("sheet", "—"),
                        "cbio_format": r.get("cbio_target_file", "—"),
                        "curability": r.get("curability", "NO"),
                        "priority": r.get("priority", "N/A"),
                        "confidence": r.get("confidence", 0),
                        "verdict": r.get("verdict", ""),
                        "req_present": r.get("required_present", []),
                        "req_missing": r.get("required_missing", []),
                        "opt_present": r.get("optional_present", []),
                    }
                    for r in records
                ],
            }

        except Exception as exc:
            st.error(f"Curation failed: {exc}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            st.stop()
        finally:
            _safe_cleanup(pdf_tmp or "", *supp_tmps)

        st.success("Curation complete.")
        st.divider()
        _render_inline_report(meta, summary)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — File Classification
# ═════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.subheader("File Classification")
    st.markdown(
        "Upload one supplementary file to detect which cBioPortal format it most closely matches."
    )

    detect_file = st.file_uploader(
        "File to classify",
        type=["xlsx", "xls", "csv", "tsv", "txt", "tab", "maf"],
        key="detect_file",
    )
    use_ai = st.checkbox("Use AI for ambiguous files", value=True, key="use_ai_detection")

    if st.button("Classify File", disabled=detect_file is None):
        try:
            from cbio_detector import detect_file_type
            from file_parser import parse_file
        except Exception as exc:
            st.error(f"Could not load classification modules: {exc}")
            st.stop()

        with st.spinner("Parsing file..."):
            try:
                df = parse_file(detect_file.getvalue(), detect_file.name)
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
                st.stop()

        st.markdown("#### File Preview")
        st.dataframe(df.head(10), use_container_width=True)

        api_key = _get_api_key() if use_ai else None
        with st.spinner("Classifying file..."):
            try:
                result = detect_file_type(df, anthropic_api_key=api_key)
            except Exception as exc:
                st.error(f"Classification failed: {exc}")
                st.stop()

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Detected Format", result.get("type", "—"))
        col2.metric("Confidence", f"{float(result.get('confidence', 0)) * 100:.0f}%")
        col3.metric("Method", "Rule-based" if result.get("method") == "heuristic" else result.get("method", "—"))

        if result.get("reasoning"):
            st.info(result["reasoning"])
        if result.get("low_confidence"):
            st.warning("Confidence is low — please verify the detected format manually.")

        mappings = result.get("column_mappings") or {}
        if mappings:
            st.markdown("#### Suggested Column Mappings")
            st.dataframe(
                pd.DataFrame(list(mappings.items()), columns=["Original Column", "cBioPortal Column"]),
                use_container_width=True,
                hide_index=True,
            )

        try:
            from spec_match import classify_sheet

            spec_result = classify_sheet(df)
            with st.expander("Detailed classification scores"):
                st.markdown(f"**Best match:** {spec_result.format_key} ({spec_result.confidence:.1f}% confidence)")
                st.markdown(f"**Target file:** {spec_result.target_file}")
                if spec_result.required_missing:
                    st.warning("Missing required columns: " + ", ".join(spec_result.required_missing))
                if spec_result.required_present:
                    st.success("Required columns found: " + ", ".join(spec_result.required_present))
                if spec_result.all_scores:
                    st.dataframe(pd.DataFrame(spec_result.all_scores), use_container_width=True, hide_index=True)
        except Exception:
            pass
