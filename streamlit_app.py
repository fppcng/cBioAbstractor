"""
cBioAbstractor
Self-contained: no backend server required.
"""

from __future__ import annotations

import io
import os
import re
import sys
import json
import time
import shutil
import tempfile
import traceback
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Dependency check ──────────────────────────────────────────────────────────
_REQUIRED = {
    "pandas":    "pandas",
    "PyPDF2":    "PyPDF2",
    "docx":      "python-docx",
    "anthropic": "anthropic",
    "chardet":   "chardet",
    "requests":  "requests",
    "plotly":    "plotly",
    "openpyxl":  "openpyxl",
}
_missing = [pkg for mod, pkg in _REQUIRED.items()
            if not importlib.util.find_spec(mod)]

import pandas as pd

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
#   2. api_config.py in the same folder
#   3. .env file in the same folder
# ─────────────────────────────────────────────────────────────────────────────
def _load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
 
    cfg_path = os.path.join(_HERE, "api_config.py")
    if os.path.exists(cfg_path):
        try:
            ns: dict = {}
            exec(open(cfg_path).read(), ns)
            key = ns.get("ANTHROPIC_API_KEY", "")
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key
                return key
        except Exception:
            pass
 
    env_path = os.path.join(_HERE, ".env")
    if os.path.exists(env_path):
        try:
            for line in open(env_path):
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY"):
                    _, _, val = line.partition("=")
                    key = val.strip().strip('"').strip("'")
                    if key:
                        os.environ["ANTHROPIC_API_KEY"] = key
                        return key
        except Exception:
            pass
 
    return ""
 
_API_KEY = _load_api_key()
 
# ── Runtime key override (for shared/cloud deployments) ───────────────────────
if not _API_KEY:
    try:
        _API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
        if _API_KEY:
            os.environ["ANTHROPIC_API_KEY"] = _API_KEY
    except Exception:
        pass
 
# ─────────────────────────────────────────────────────────────────────────────
# Helper functions — defined BEFORE sidebar so they can be called there
# ─────────────────────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    return _API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
 
def _require_api_key() -> bool:
    if not _get_api_key():
        st.error("⬅️ Enter your Anthropic API key in the sidebar to continue.")
        return False
    return True
 
def _save_upload_to_tmp(uploaded) -> tuple[str, str]:
    original_name = uploaded.name
    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, original_name)
    with open(path, "wb") as f:
        f.write(uploaded.getvalue())
    return path, original_name
 
def _safe_cleanup(*paths: str) -> None:
    for p in paths:
        try:
            shutil.rmtree(os.path.dirname(p), ignore_errors=True)
        except Exception:
            pass
 
def _call_anthropic_with_retry(client, model, system, user_content,
                                max_tokens=2000, retries=3, backoff=5.0):
    import anthropic
    last_exc = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            return resp.content[0].text
        except anthropic.RateLimitError as e:
            time.sleep(backoff * (attempt + 1)); last_exc = e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(backoff * (attempt + 1)); last_exc = e
            else:
                raise
        except anthropic.APIConnectionError as e:
            time.sleep(backoff * (attempt + 1)); last_exc = e
    raise last_exc or RuntimeError("API call failed after retries")
 
def _parse_llm_json(raw: str) -> dict:
    raw = re.sub(r"^```[^\n]*\n?", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)
 
def _looks_tmp(name: str) -> bool:
    return bool(re.match(r'^tmp[a-z0-9_]{4,}', os.path.splitext(name)[0], re.I))
 
# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 4px 0;">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 48" width="200">
            <rect width="220" height="48" rx="6" fill="#102d4b"/>
            <text x="12" y="20" font-family="Arial,sans-serif" font-size="13"
                  font-weight="700" fill="#e8523a" letter-spacing="0.5">cBio</text>
            <text x="46" y="20" font-family="Arial,sans-serif" font-size="13"
                  font-weight="700" fill="#ffffff" letter-spacing="0.5">Portal</text>
            <text x="12" y="38" font-family="Arial,sans-serif" font-size="10"
                  fill="#7eb8d4" letter-spacing="1">FOR CANCER GENOMICS</text>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Automated curation support for cancer genomics studies.")
    st.divider()
 
    if _missing:
        st.warning("Some packages are installing. Please refresh in 1–2 minutes.")
 
    if _get_api_key():
        st.success("Connected")
    else:
        st.warning("API key not configured. Contact your administrator.")
 
    st.divider()
    st.caption("Version 1.1")
 
# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.markdown(
        """
        <div style="padding: 12px 0 0 0;">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 54" width="180">
            <rect width="220" height="54" rx="7" fill="#102d4b"/>
            <text x="14" y="23" font-family="Arial,sans-serif" font-size="15"
                  font-weight="700" fill="#e8523a" letter-spacing="0.5">cBio</text>
            <text x="52" y="23" font-family="Arial,sans-serif" font-size="15"
                  font-weight="700" fill="#ffffff" letter-spacing="0.5">Portal</text>
            <text x="14" y="42" font-family="Arial,sans-serif" font-size="10"
                  fill="#7eb8d4" letter-spacing="1">FOR CANCER GENOMICS</text>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_title:
    st.title("Data Abstractor")
    st.markdown(
        "Upload a published cancer genomics paper and its supplementary data files "
        "to generate a structured curation summary for cBioPortal ingestion."
    )
 
# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_curate, tab_detect = st.tabs([
    "Curation Report",
    "File Classification",
])
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers for the report table
# ─────────────────────────────────────────────────────────────────────────────
 
def _colour_curability(val):
    return {
        "Yes":                      "background-color:#E2EFDA;color:#375623",
        "Partly curatable":           "background-color:#FFF2CC;color:#7F6000",
        "Needs manual intervention": "background-color:#FCE4D6;color:#843C0C",
    }.get(val, "")
 
def _colour_priority(val):
    return {
        "HIGH":   "background-color:#FCE4D6;color:#843C0C",
        "MEDIUM": "background-color:#FFF2CC;color:#7F6000",
        "LOW":    "background-color:#E2EFDA;color:#375623",
        "N/A":    "background-color:#F2F2F2;color:#595959",
    }.get(val, "")
 
def _colour_confidence(val):
    try:
        v = float(str(val).replace("%", ""))
        if v >= 70: return "background-color:#E2EFDA;color:#375623"
        if v >= 40: return "background-color:#FFF2CC;color:#7F6000"
        return "background-color:#FCE4D6;color:#843C0C"
    except Exception:
        return ""
 
def _render_inline_report(meta: dict, records: list, summary: dict):
    """Render the full curation report as inline Streamlit content."""
 
    def _curability_label(val: str) -> str:
        return {
            "YES":     "Yes",
            "PARTIAL": "Partly curatable",
            "NO":      "Needs manual intervention",
        }.get(val, val)
 
    # ── Study overview ────────────────────────────────────────────────────────
    st.markdown("## Study Overview")
 
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Study ID",        summary.get("study_id") or "—")
    ov2.metric("Cancer Type",     summary.get("cancer_type") or "—")
    ov3.metric("Samples",         summary.get("num_samples") or "—")
    ov4.metric("Reference Genome", summary.get("reference_genome") or "—")
 
    if meta.get("study_title"):
        st.markdown(f"**Title:** {meta['study_title']}")
    if meta.get("cancer_type_full"):
        st.markdown(f"**Cancer type (full):** {meta['cancer_type_full']}")
    if meta.get("primary_site"):
        st.markdown(f"**Primary site:** {meta['primary_site']}")
    if meta.get("journal") or meta.get("year"):
        st.markdown(
            f"**Publication:** {meta.get('journal', '')} {meta.get('year', '')}".strip()
        )
    if meta.get("pmid"):
        st.markdown(f"**PMID:** {meta['pmid']}")
    if meta.get("doi"):
        st.markdown(f"**DOI:** {meta['doi']}")
    if meta.get("first_author_surname"):
        st.markdown(f"**First author:** {meta['first_author_surname']}")
    if meta.get("corresponding_authors"):
        st.markdown(f"**Corresponding author(s):** {meta['corresponding_authors']}")
    if meta.get("cohort_description"):
        st.markdown(f"**Cohort:** {meta['cohort_description']}")
    if meta.get("sequencing_types"):
        seq = meta["sequencing_types"]
        if isinstance(seq, list):
            seq = ", ".join(seq)
        st.markdown(f"**Sequencing:** {seq}")
    if meta.get("data_repositories"):
        repos = meta["data_repositories"]
        if isinstance(repos, list):
            repos = ", ".join(repos)
        st.markdown(f"**Data repositories:** {repos}")
    if meta.get("description"):
        st.markdown(f"**Summary:** {meta['description']}")
    if meta.get("key_findings"):
        st.markdown("**Key findings:**")
        for kf in meta["key_findings"]:
            st.markdown(f"- {kf}")
 
    st.divider()
 
    # ── Supplementary file analysis ───────────────────────────────────────────
    st.markdown("## Supplementary File Analysis")
 
    pr1, pr2, pr3 = st.columns(3)
    pr1.metric("High Priority",   summary.get("high_priority", 0))
    pr2.metric("Medium Priority", summary.get("medium_priority", 0))
    pr3.metric("Needs Manual Intervention", summary.get("not_loadable", 0))
 
    def _fmt_label(val: str) -> str:
        """Clean up cBioPortal format labels for display."""
        replacements = {
            "Not directly loadable": "Needs manual intervention",
            "NOT_LOADABLE":          "Needs manual intervention",
        }
        return replacements.get(val, val)
 
    breakdown = summary.get("file_breakdown", [])
    if breakdown:
        df_bd = pd.DataFrame([{
            "File":              row["file"],
            "Sheet":             row["sheet"],
            "cBioPortal Format": _fmt_label(row["cbio_format"]),
            "Confidence":        f"{row.get('confidence', 0):.0f}%",
            "Loadable":          _curability_label(row["curability"]),
            "Priority":          row["priority"],
            "Columns Present":   ", ".join(row.get("req_present", [])) or "—",
            "Columns Missing":   ", ".join(row.get("req_missing", [])) or "None",
        } for row in breakdown])
 
        styled = (
            df_bd.style
            .map(_colour_curability, subset=["Loadable"])
            .map(_colour_priority,   subset=["Priority"])
            .map(_colour_confidence, subset=["Confidence"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
 
    st.divider()
 
    # ── Per-sheet detail ──────────────────────────────────────────────────────
    st.markdown("## Per-Sheet Classification Detail")
    for row in breakdown:
        label = "{} — {}".format(row["file"], row["sheet"])
        with st.expander(label, expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Format:** {_fmt_label(row['cbio_format'])}")
            c2.markdown(f"**Confidence:** {row.get('confidence', 0):.0f}%")
            c3.markdown(f"**Priority:** {row['priority']}")
 
            if row.get("verdict"):
                st.markdown(f"**Assessment:** {row['verdict']}")
            if row.get("req_present"):
                st.success("Required columns found: " + ", ".join(row["req_present"]))
            if row.get("req_missing"):
                st.warning("Required columns missing: " + ", ".join(row["req_missing"]))
            if row.get("opt_present"):
                st.info("Optional columns found: " + ", ".join(row["opt_present"]))
 
    st.divider()
 
    # ── Curation checklist ────────────────────────────────────────────────────
    st.markdown("## Curation Checklist")
 
    high_items  = [r for r in breakdown if r.get("priority") == "HIGH"  and r.get("curability") != "NO"]
    med_items   = [r for r in breakdown if r.get("priority") == "MEDIUM" and r.get("curability") != "NO"]
    skip_items  = [r for r in breakdown if r.get("curability") == "NO"]
 
    if high_items:
        st.markdown("### High Priority")
        for r in high_items:
            st.markdown("- **{} / {}** → `{}`{}".format(
                r["file"], r["sheet"], _fmt_label(r["cbio_format"]),
                " *(missing: {})*".format(", ".join(r.get("req_missing", []))) if r.get("req_missing") else ""
            ))
        st.markdown("### Medium Priority")
        for r in med_items:
            st.markdown("- **{} / {}** → `{}`{}".format(
                r["file"], r["sheet"], _fmt_label(r["cbio_format"]),
                " *(missing: {})*".format(", ".join(r.get("req_missing", []))) if r.get("req_missing") else ""
            ))
 
    if skip_items:
        st.markdown("### Needs Manual Intervention")
        for r in skip_items:
            st.markdown("- {} / {}".format(r["file"], r["sheet"]))
 
    st.divider()
 
    # ── Study metadata for meta files ─────────────────────────────────────────
    st.markdown("## Suggested Study Metadata")
    st.markdown("Use these values when creating your `meta_study.txt` and `meta_cancer_type.txt` files.")
 
    meta_rows = {
        "cancer_study_identifier": summary.get("study_id") or "—",
        "name":                    meta.get("study_title") or "—",
        "description":             meta.get("meta_description") or meta.get("description") or "—",
        "cancer_type":             meta.get("cancer_type") or "—",
        "short_name":              meta.get("study_id_suggestion") or "—",
        "pmid":                    meta.get("pmid") or "—",
        "groups":                  "PUBLIC",
    }
    meta_df = pd.DataFrame(
        [{"Field": k, "Value": v} for k, v in meta_rows.items()]
    )
    st.dataframe(meta_df, use_container_width=True, hide_index=True)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Curation Report
# ═════════════════════════════════════════════════════════════════════════════
with tab_curate:
    st.subheader("Curation Report Generator")
    st.markdown(
        "Upload the published paper PDF and its supplementary data files. "
        "The tool will extract study metadata, classify each data file against "
        "cBioPortal format requirements, and produce a structured curation summary."
    )
 
    # Spec freshness indicator
    try:
        from spec_fetcher import fetch_spec
        _spec = fetch_spec()
        if _spec.get("source") == "live":
            st.caption(
                f"Format specifications loaded from cBioPortal documentation "
                f"({len(_spec.get('specs', []))} formats)."
            )
        else:
            st.caption("Using embedded format specifications (offline mode).")
    except Exception:
        pass
 
    st.divider()
 
    col_pdf, col_supp = st.columns(2)
    with col_pdf:
        st.markdown("#### Paper PDF")
        paper_pdf = st.file_uploader(
            "Select the main paper PDF",
            type=["pdf"],
            key="cbio_paper_pdf",
        )
 
    confirmed_names: list[str] = []  # populated below when files are uploaded
 
    with col_supp:
        st.markdown("#### Supplementary Files")
        supp_files = st.file_uploader(
            "Select supplementary data files",
            type=["xlsx", "xls", "csv", "tsv", "txt", "tab", "maf", "doc", "docx", "pdf"],
            accept_multiple_files=True,
            key="cbio_supp_files",
            help="Accepts Excel, CSV, TSV, MAF, and Word files.",
        )
 
    # ── Filename inputs ───────────────────────────────────────────────────────
    # Always show name fields when files are uploaded. Names are written to
    # session_state immediately and read back at button-click time, so they
    # survive the Streamlit re-run that the button triggers.
    if "fname_count" not in st.session_state:
        st.session_state["fname_count"] = 0
 
    if supp_files:
        # Update session_state with current file list on every render
        st.session_state["fname_count"] = len(supp_files)
        import re as _re_name
        def _looks_tmp(n):
            return bool(_re_name.match(r'^tmp[a-z0-9_]{4,}', os.path.splitext(n)[0], _re_name.I))
 
        st.divider()
        st.markdown("#### File Names")
        st.caption("Auto-filled below — edit if any name is wrong before clicking Generate.")
        cols = st.columns(min(len(supp_files), 3))
        for i, sf in enumerate(supp_files):
            ext  = os.path.splitext(sf.name)[1] or ".xlsx"
            auto = f"Supplementary_Data_{i+1}{ext}"
            # Only auto-fill if current stored value looks like a temp name or is missing
            current = st.session_state.get(f"fname_{i}", "")
            if not current or _looks_tmp(current):
                default = auto if _looks_tmp(sf.name) else sf.name
            else:
                default = current
            val = cols[i % len(cols)].text_input(
                f"File {i+1}", value=default, key=f"fname_widget_{i}"
            )
            st.session_state[f"fname_{i}"] = val.strip() or auto
 
    st.divider()
 
    with st.expander("Options"):
        llm_model = st.selectbox(
            "AI model",
            options=[
                "anthropic/claude-sonnet-4-20250514",
                "anthropic/claude-3-5-haiku-20241022",
                "openai/gpt-4o",
                "openai/gpt-4-turbo",
            ],
            key="cbio_llm_model",
        )
        temperature = st.slider(
            "Response variability (0 = consistent, 1 = creative)",
            0.0, 1.0, 0.2, 0.05,
            key="cbio_temp",
        )
 
    if st.button(
        "Generate Curation Report",
        disabled=(paper_pdf is None),
        type="primary",
        key="cbio_run_btn",
    ):
        if not _require_api_key():
            st.stop()
 
        # ── Save uploads ──────────────────────────────────────────────────────
        pdf_tmp    = None
        supp_tmps: list[str] = []
        try:
            with st.spinner("Saving uploaded files..."):
                pdf_tmp, _ = _save_upload_to_tmp(paper_pdf)
                supp_list  = list(supp_files or [])
 
                # Read names from session_state (survive button re-run)
                n = st.session_state.get("fname_count", len(supp_list))
                orig_names = [
                    st.session_state.get(f"fname_{i}", f"Supplementary_Data_{i+1}.xlsx")
                    for i in range(n)
                ]
                orig_names = orig_names[:len(supp_list)]
                while len(orig_names) < len(supp_list):
                    orig_names.append(f"Supplementary_Data_{len(orig_names)+1}.xlsx")
 
                for sf, orig in zip(supp_list, orig_names):
                    tmp_dir = tempfile.mkdtemp()
                    dest    = os.path.join(tmp_dir, orig)
                    with open(dest, "wb") as fh:
                        fh.write(sf.getvalue())
                    supp_tmps.append(dest)
 
            use_anthropic   = llm_model.startswith("anthropic/")
            anthropic_model = llm_model.split("/", 1)[1] if use_anthropic else None
 
            # ── Step 1: Extract metadata ──────────────────────────────────────
            meta: dict = {}
            with st.spinner("Step 1 of 2 — Extracting study metadata from PDF..."):
                if use_anthropic:
                    import anthropic as _anthropic
                    from cbioportal_curator import _extract_pdf_text, SYSTEM_PROMPT_CURATOR
 
                    pdf_text = _extract_pdf_text(pdf_tmp)
                    if not pdf_text.strip():
                        st.warning("Could not extract text from the PDF. Metadata fields will be blank.")
                    else:
                        _client = _anthropic.Anthropic(api_key=_get_api_key())
                        try:
                            raw_meta = _call_anthropic_with_retry(
                                _client,
                                model=anthropic_model,
                                system=SYSTEM_PROMPT_CURATOR,
                                user_content=pdf_text[:40000],
                                max_tokens=2000,
                            )
                            meta = _parse_llm_json(raw_meta)
                        except json.JSONDecodeError:
                            st.warning("Metadata extraction returned unexpected format. Continuing with file classification.")
                            meta = {}
                        except Exception as e:
                            st.warning(f"Metadata extraction failed ({e}). Continuing with file classification.")
                            meta = {}
 
            # ── Step 2: Classify supplementary files ─────────────────────────
            records: list[dict] = []
            with st.spinner(f"Step 2 of 2 — Classifying {len(supp_tmps)} supplementary file(s)..."):
                if use_anthropic:
                    from cbioportal_curator import _analyse_supplementary_files
 
                    try:
                        records = _analyse_supplementary_files(supp_tmps)
                    except Exception as e:
                        st.error(f"File classification failed: {e}")
                        with st.expander("Error details"):
                            st.code(traceback.format_exc())
                        st.stop()
 
                    # Positional filename patch
                    _by_file: dict[str, list] = {}
                    for rec in records:
                        _by_file.setdefault(rec.get("file", ""), []).append(rec)
                    for i, orig in enumerate(orig_names):
                        key = list(_by_file.keys())[i] if i < len(_by_file) else None
                        if key and key != orig:
                            for rec in _by_file[key]:
                                rec["file"] = orig
 
                else:
                    from cbioportal_curator import curate
                    result  = curate(
                        pdf_path=pdf_tmp, supp_paths=supp_tmps,
                        llm_model=llm_model, temperature=temperature,
                    )
                    meta    = meta or {}
                    records = []
                    summary = result["summary"]
 
            # ── Build summary ─────────────────────────────────────────────────
            if use_anthropic:
                summary = {
                    "study_id":         meta.get("study_id_suggestion") or "—",
                    "cancer_type":      meta.get("cancer_type") or "—",
                    "num_samples":      meta.get("num_samples") or "—",
                    "reference_genome": meta.get("reference_genome") or "—",
                    "files_analysed":   len(supp_tmps),
                    "sheets_analysed":  len(records),
                    "high_priority":    sum(1 for r in records if r.get("priority") == "HIGH"),
                    "medium_priority":  sum(1 for r in records if r.get("priority") == "MEDIUM"),
                    "not_loadable":     sum(1 for r in records if r.get("curability") == "NO"),
                    "file_breakdown": [{
                        "file":        r["file"],
                        "sheet":       r["sheet"],
                        "cbio_format": r.get("cbio_target_file", "—"),
                        "curability":  r.get("curability", "NO"),
                        "priority":    r.get("priority", "N/A"),
                        "confidence":  r.get("confidence", 0),
                        "verdict":     r.get("verdict", ""),
                        "req_present": r.get("required_present", []),
                        "req_missing": r.get("required_missing", []),
                        "opt_present": r.get("optional_present", []),
                    } for r in records],
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
        _render_inline_report(meta, records, summary)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — File Classification
# ═════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.subheader("File Classification")
    st.markdown(
        "Upload a single supplementary data file to identify which cBioPortal "
        "format it corresponds to. The tool checks column names and data patterns "
        "against the cBioPortal specification, then uses AI for ambiguous cases."
    )
 
    detect_file = st.file_uploader(
        "Select a file to classify",
        type=["xlsx", "xls", "csv", "tsv", "txt", "maf"],
        key="detect_file",
    )
    use_ai = st.checkbox(
        "Use AI for ambiguous files (requires API key)", value=True, key="det_use_ai"
    )
 
    if st.button("Classify File", disabled=(detect_file is None), key="detect_btn"):
        from file_parser import parse_file
        from cbio_detector import detect_file_type
 
        with st.spinner("Parsing file..."):
            try:
                df = parse_file(detect_file.getvalue(), detect_file.name)
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
                st.stop()
 
        st.markdown("#### File Preview")
        st.dataframe(df.head(10), use_container_width=True)
 
        akey = _get_api_key() if use_ai else None
        with st.spinner("Classifying..."):
            try:
                result = detect_file_type(df, anthropic_api_key=akey)
            except Exception as exc:
                st.error(f"Classification failed: {exc}")
                st.stop()
 
        st.divider()
        ca, cb, cc = st.columns(3)
        ca.metric("Detected Format",   result["type"])
        cb.metric("Confidence",        f"{result['confidence']*100:.0f}%")
        cc.metric("Method",
                  "Rule-based" if result["method"] == "heuristic" else "AI-assisted")
 
        if result.get("reasoning"):
            st.info(result["reasoning"])
        if result.get("low_confidence"):
            st.warning("Confidence is low — please verify the detected format manually.")
 
        if result.get("column_mappings"):
            st.markdown("#### Suggested Column Mappings")
            st.dataframe(
                pd.DataFrame(
                    list(result["column_mappings"].items()),
                    columns=["Original Column", "cBioPortal Column"],
                ),
                use_container_width=True, hide_index=True,
            )
 
        # Spec-based detail
        try:
            from spec_match import classify_sheet
            sr = classify_sheet(df)
            with st.expander("Detailed classification scores"):
                st.markdown(f"**Best match:** {sr.format_key} ({sr.confidence:.1f}% confidence)")
                st.markdown(f"**Target file:** {sr.target_file}")
                if sr.required_missing:
                    st.warning("Missing required columns: " + ", ".join(sr.required_missing))
                if sr.required_present:
                    st.success("Required columns found: " + ", ".join(sr.required_present))
                if sr.all_scores:
                    st.dataframe(pd.DataFrame(sr.all_scores),
                                 use_container_width=True, hide_index=True)
        except Exception:
            pass
