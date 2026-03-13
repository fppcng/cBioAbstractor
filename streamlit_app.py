import os
import streamlit as st
import requests
import pandas as pd
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Render free tier cold-starts can take 30-60 s.
# These timeouts are intentionally generous.
TIMEOUT_FAST   = 60    # simple pings / downloads
TIMEOUT_NORMAL = 300   # file uploads + LLM calls
TIMEOUT_LONG   = 600   # curation report (PDF + many Excel sheets + LLM)


def _wake_backend(placeholder):
    """
    Ping the API root until it responds (handles Render cold-start).
    Shows a live status message in *placeholder* while waiting.
    Returns True if the backend is up, False if it never responded.
    """
    for attempt in range(1, 19):          # up to ~90 s
        try:
            r = requests.get(API_URL + "/", timeout=10)
            if r.status_code < 500:
                placeholder.empty()
                return True
        except requests.exceptions.RequestException:
            pass
        placeholder.info(
            f"⏳ Waking up the server… ({attempt * 5} s elapsed)  \n"
            "The backend may be starting from sleep — this usually takes 30–90 s on Render free tier."
        )
        time.sleep(5)
    placeholder.error(
        "❌ Backend did not respond after 90 s. "
        "Check that the Render service is running."
    )
    return False


# ─────────────────────────────────────────────────────────────
# Helper: render a single code Q&A response
# ─────────────────────────────────────────────────────────────

def _render_qa_result(result: dict):
    """Render a /code_query/ response dict in the chat."""
    result_type = result.get("result_type", "text")
    answer      = result.get("answer")
    explanation = result.get("explanation", "")
    code        = result.get("code", "")
    error       = result.get("error")

    # Answer
    if result_type == "dataframe" and isinstance(answer, list) and answer:
        st.dataframe(pd.DataFrame(answer), use_container_width=True, hide_index=True)
    elif result_type == "dict" and isinstance(answer, dict):
        for k, v in answer.items():
            st.write(f"**{k}:** {v}")
    elif result_type == "list" and isinstance(answer, list):
        for item in answer:
            st.write(f"- {item}")
    else:
        st.write(str(answer))

    # Explanation
    if explanation:
        st.caption(explanation)

    # Code expander
    if code:
        with st.expander("🔍  View generated code"):
            st.code(code, language="python")

    # Error display
    if error:
        with st.expander("⚠️  Execution error"):
            st.code(error, language="text")

st.set_page_config(
    page_title="Synopsis — Literature & cBioPortal Curator",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Synopsis")
st.caption(
    "Literature Retrieval Evidence Summarization  ·  "
    "cBioPortal Curation  ·  Gene Alteration Analysis"
)

tab_lit, tab_cbio, tab_gene = st.tabs([
    "📚  Literature Retrieval",
    "🗂️  cBioPortal Curator",
    "🧪  Gene Alterations & Code Q&A",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — Literature Retrieval  (unchanged)
# ═══════════════════════════════════════════════════════════════════════

with tab_lit:
    with st.sidebar:
        st.title("Vector Store — PDF Loader")
        uploaded_files = st.file_uploader(
            "Choose PDF files", accept_multiple_files=True, type="pdf"
        )
        if st.button("Load PDFs", key="load_pdfs_button"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    response = requests.post(
                        f"{API_URL}/ingest_pdf/",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    )
                    if response.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} ingested.")
                    else:
                        st.error(f"❌ Error processing {uploaded_file.name}.")
                st.success("Done")
            else:
                st.warning("Please upload at least one PDF.")

        if st.button("Clear Vector Store", key="clear_vector_store_button"):
            response = requests.post(f"{API_URL}/clear_vector_store/")
            if response.status_code == 200:
                st.success("Vector store cleared.")
            else:
                st.error("Error clearing vector store.")

    st.subheader("Ask a question")
    question = st.text_input("Provide parameters to generate evidence-based answers")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Searching vector store and generating answer…"):
                response = requests.post(
                    f"{API_URL}/generate_evidence/", data={"question": question}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.write("**Answer:**")
                    st.write(data.get("answer", "No answer returned."))
                else:
                    st.error("Error fetching answer.")
        else:
            st.warning("Please enter a question.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — cBioPortal Curator  (unchanged)
# ═══════════════════════════════════════════════════════════════════════

with tab_cbio:
    st.subheader("cBioPortal Data Curation Report Generator")
    st.markdown(
        """
        Upload the **main paper PDF** and any **supplementary files** from a
        cancer genomics study.
        The tool will:
        - Extract study metadata (cancer type, cohort size, reference genome, PMID…)
        - Classify every supplementary sheet against cBioPortal file formats
        - Build column-mapping & transformation instructions
        - Generate a downloadable **.docx** curation report
        """
    )

    # Spec status indicator
    try:
        spec_info = requests.get(f"{API_URL}/spec_status", timeout=10)
        if spec_info.status_code == 200:
            si = spec_info.json()
            if si.get("source") == "live":
                st.success(
                    f"✅ Format spec: **live** from docs.cbioportal.org  "
                    f"({si.get('num_formats', '?')} formats)  "
                    f"— fetched {si.get('fetched_at','?')[:19].replace('T',' ')} UTC"
                )
            else:
                st.warning(
                    "⚠️ Format spec: **embedded fallback** "
                    f"(live fetch unavailable: {si.get('error','')})"
                )
    except Exception:
        pass   # Silently skip if backend is cold

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1.  Upload Paper PDF")
        paper_pdf = st.file_uploader("Main paper (PDF)", type=["pdf"], key="cbio_paper_pdf")

    with col2:
        st.markdown("#### 2.  Upload Supplementary Files")
        supp_files = st.file_uploader(
            "Supplementary data files (.xlsx, .csv, .tsv, .txt, .maf, .docx, .pdf)",
            type=["xlsx", "xls", "csv", "tsv", "txt", "tab", "maf", "doc", "docx", "pdf"],
            accept_multiple_files=True,
            key="cbio_supp_files",
        )

    st.divider()

    with st.expander("⚙️  Advanced Options"):
        llm_model = st.selectbox(
            "LLM for metadata extraction",
            options=[
                "openai/gpt-4o",
                "openai/gpt-4-turbo",
                "openai/gpt-3.5-turbo",
                "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
                "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            ],
        )
        temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)

    if st.button("🚀  Generate cBioPortal Curation Report",
                 disabled=(paper_pdf is None), type="primary"):
        wake_ph = st.empty()
        if not _wake_backend(wake_ph):
            st.stop()

        with st.spinner("Analysing paper and supplementary files… (may take 1–3 min)"):
            files_payload = [
                ("paper_pdf", (paper_pdf.name, paper_pdf.getvalue(), "application/pdf"))
            ]
            for sf in (supp_files or []):
                files_payload.append((
                    "supplementary_files",
                    (sf.name, sf.getvalue(),
                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                ))
            try:
                resp = requests.post(
                    f"{API_URL}/curate_cbioportal/",
                    files=files_payload,
                    data={"llm_model": llm_model, "temperature": str(temperature)},
                    timeout=TIMEOUT_LONG,
                )
            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ Request timed out after 10 min. "
                    "Try uploading fewer supplementary files, or check the server logs."
                )
                st.stop()
            except requests.exceptions.ConnectionError as e:
                st.error(f"🔌 Connection error: {e}")
                st.stop()

        if resp.status_code == 200:
            result = resp.json()
            st.success("✅ Curation report generated!")
            st.divider()

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Study ID",        result.get("study_id", "—"))
            m2.metric("Cancer Type",     result.get("cancer_type", "—"))
            m3.metric("Samples",         result.get("num_samples", "—"))
            m4.metric("Files Analysed",  result.get("files_analysed", "—"))
            m5.metric("Sheets Analysed", result.get("sheets_analysed", "—"))

            p1, p2, p3 = st.columns(3)
            p1.metric("🔴 High Priority",   result.get("high_priority", 0))
            p2.metric("🟡 Medium Priority", result.get("medium_priority", 0))
            p3.metric("⬜ Not Loadable",     result.get("not_loadable", 0))

            st.divider()
            st.markdown("### Supplementary File Breakdown")
            breakdown = result.get("file_breakdown", [])
            if breakdown:
                def _style_cur(val):
                    c = {"YES": "#E2EFDA;color:#375623",
                         "PARTIAL": "#FFF2CC;color:#7F6000",
                         "NO": "#FCE4D6;color:#843C0C"}
                    return f"background-color:{c[val]}" if val in c else ""

                def _style_pri(val):
                    c = {"HIGH": "#FCE4D6;color:#843C0C",
                         "MEDIUM": "#FFF2CC;color:#7F6000",
                         "LOW": "#E2EFDA;color:#375623",
                         "N/A": "#F2F2F2;color:#595959"}
                    return f"background-color:{c[val]}" if val in c else ""

                def _style_conf(val):
                    try:
                        v = float(str(val).replace("%", ""))
                        if v >= 70: return "background-color:#E2EFDA;color:#375623"
                        if v >= 40: return "background-color:#FFF2CC;color:#7F6000"
                        return "background-color:#FCE4D6;color:#843C0C"
                    except Exception:
                        return ""

                df_bd = pd.DataFrame([{
                    "File":              row["file"],
                    "Sheet":             row["sheet"],
                    "cBioPortal Format": row["cbio_format"],
                    "Confidence":        f"{row.get('confidence', 0):.0f}%",
                    "Curate?":           row["curability"],
                    "Priority":          row["priority"],
                    "Required ✓":        ", ".join(row.get("req_present", [])) or "—",
                    "Required ✗":        ", ".join(row.get("req_missing", [])) or "none",
                } for row in breakdown])

                styled = (df_bd.style
                          .applymap(_style_cur,  subset=["Curate?"])
                          .applymap(_style_pri,  subset=["Priority"])
                          .applymap(_style_conf, subset=["Confidence"]))
                st.dataframe(styled, use_container_width=True, hide_index=True)

                with st.expander("🔍  Classification verdicts (per sheet)"):
                    for row in breakdown:
                        st.markdown(
                            f"**{row['file']} — {row['sheet']}**  \n"
                            f"{row.get('verdict', '')}"
                        )
                        if row.get("req_missing"):
                            st.caption(
                                "⚠️ Missing required: "
                                + ", ".join(row["req_missing"])
                            )

            st.divider()
            dl_url = result.get("report_download_url")
            if dl_url:
                dl = requests.get(f"{API_URL}{dl_url}", timeout=TIMEOUT_FAST)
                if dl.status_code == 200:
                    st.download_button(
                        "📥  Download cBioPortal Curation Report (.docx)",
                        data=dl.content,
                        file_name=result.get("report_filename",
                                             "cbioportal_curation_report.docx"),
                        mime=("application/vnd.openxmlformats-officedocument"
                              ".wordprocessingml.document"),
                        type="primary",
                    )
        else:
            st.error(f"API error ({resp.status_code}): {resp.text[:400]}")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — Gene Alteration Analysis + Code Q&A  (new)
# ═══════════════════════════════════════════════════════════════════════

with tab_gene:
    st.subheader("Gene Alteration Frequencies & Code Q&A")
    st.markdown(
        """
        Upload a **MAF, Excel, or CSV** genomic data file to:
        - Compute per-gene **mutation, amplification, deletion, SV, and combined**
          alteration frequencies (% samples altered)
        - Ask free-text questions about the data — the tool writes and runs Python
          code to answer them
        """
    )

    st.divider()

    # ── 1. File upload & frequency computation ──────────────────────────
    st.markdown("#### 1.  Upload Genomic Data File")
    st.caption(
        "Accepted formats: MAF (.maf / .txt / .tsv), Excel (.xlsx), CSV (.csv).  "
        "Automatically detects mutations, CNA matrices, and SV/fusion tables."
    )
    gene_file = st.file_uploader(
        "Choose file",
        type=["maf", "txt", "tsv", "csv", "xlsx"],
        key="gene_data_file",
    )

    if "gene_session_id" not in st.session_state:
        st.session_state["gene_session_id"] = None
    if "gene_freq_df" not in st.session_state:
        st.session_state["gene_freq_df"] = None
    if "gene_summary" not in st.session_state:
        st.session_state["gene_summary"] = None
    if "code_qa_history" not in st.session_state:
        st.session_state["code_qa_history"] = []

    load_btn = st.button(
        "📊  Compute Alteration Frequencies",
        disabled=(gene_file is None),
        type="primary",
    )

    if load_btn and gene_file:
        wake_ph = st.empty()
        if not _wake_backend(wake_ph):
            st.stop()

        with st.spinner("Parsing file and computing frequencies…"):
            try:
                resp = requests.post(
                    f"{API_URL}/gene_alterations/",
                    files={"data_file": (gene_file.name, gene_file.getvalue())},
                    timeout=TIMEOUT_NORMAL,
                )
            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ Request timed out after 5 min. "
                    "The file may be very large — try a smaller subset."
                )
                st.stop()
            except requests.exceptions.ConnectionError as e:
                st.error(f"🔌 Connection error: {e}")
                st.stop()
        if resp.status_code == 200:
            payload = resp.json()
            st.session_state["gene_session_id"] = payload["session_id"]
            st.session_state["gene_summary"]    = payload["summary"]
            st.session_state["code_qa_history"] = []  # reset chat on new file
            freq_records = payload.get("frequencies", [])
            if freq_records:
                st.session_state["gene_freq_df"] = pd.DataFrame(freq_records)
            else:
                st.session_state["gene_freq_df"] = pd.DataFrame()
            st.success("✅ File loaded successfully!")
        else:
            st.error(f"Error ({resp.status_code}): {resp.text[:400]}")

    # ── 2. Summary metrics & visualisation ─────────────────────────────
    if st.session_state["gene_summary"]:
        summ = st.session_state["gene_summary"]
        st.divider()
        st.markdown("### Dataset Summary")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Samples",   summ.get("n_samples", "—"))
        c2.metric("Genes",     summ.get("n_genes",   "—"))
        c3.metric("Mutations", "✓" if summ.get("has_mutations") else "—")
        c4.metric("CNA",       "✓" if summ.get("has_cna")       else "—")
        c5.metric("SV/Fusion", "✓" if summ.get("has_sv")        else "—")

        freq_df: pd.DataFrame = st.session_state["gene_freq_df"]

        if freq_df is not None and not freq_df.empty:
            st.divider()

            # ── Controls row ──────────────────────────────────────────
            ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
            with ctrl1:
                top_n = st.slider("Top N genes to display", 5, 50, 20, 5,
                                  key="top_n_slider")
            with ctrl2:
                sort_col = st.selectbox(
                    "Sort by",
                    ["pct_any", "pct_mutated", "pct_amp", "pct_del", "pct_sv"],
                    key="sort_col_select",
                )
            with ctrl3:
                chart_type = st.selectbox(
                    "Chart type",
                    ["Stacked bar", "Side-by-side bar", "Table only"],
                    key="chart_type_select",
                )

            # Prepare display slice
            gene_col = "gene" if "gene" in freq_df.columns else freq_df.columns[0]
            display_df = (
                freq_df.sort_values(sort_col, ascending=False)
                       .head(top_n)
            )

            # ── Chart ─────────────────────────────────────────────────
            if chart_type != "Table only":
                try:
                    import plotly.graph_objects as go

                    genes    = display_df[gene_col].tolist()
                    mut_vals = display_df.get("pct_mutated",
                               pd.Series([0]*len(genes))).fillna(0).tolist()
                    amp_vals = display_df.get("pct_amp",
                               pd.Series([0]*len(genes))).fillna(0).tolist()
                    del_vals = display_df.get("pct_del",
                               pd.Series([0]*len(genes))).fillna(0).tolist()
                    sv_vals  = display_df.get("pct_sv",
                               pd.Series([0]*len(genes))).fillna(0).tolist()

                    bar_mode = "stack" if chart_type == "Stacked bar" else "group"
                    colour   = {
                        "Mutation":      "#E05C5C",
                        "Amplification": "#F5A623",
                        "Deletion":      "#4A90D9",
                        "SV/Fusion":     "#7ED321",
                    }

                    fig = go.Figure()
                    for label, vals, col in [
                        ("Mutation",      mut_vals, colour["Mutation"]),
                        ("Amplification", amp_vals, colour["Amplification"]),
                        ("Deletion",      del_vals, colour["Deletion"]),
                        ("SV/Fusion",     sv_vals,  colour["SV/Fusion"]),
                    ]:
                        if any(v > 0 for v in vals):
                            fig.add_trace(go.Bar(
                                name=label, x=genes, y=vals,
                                marker_color=col,
                                hovertemplate="%{x}<br>"
                                              + label + ": %{y:.1f}%<extra></extra>",
                            ))

                    fig.update_layout(
                        barmode=bar_mode,
                        xaxis_title="Gene",
                        yaxis_title="% Samples Altered",
                        yaxis=dict(range=[0, 100]),
                        legend=dict(orientation="h", yanchor="bottom",
                                    y=1.02, xanchor="right", x=1),
                        height=420,
                        margin=dict(t=40, b=60),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                    )
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                except ImportError:
                    st.warning(
                        "plotly not installed — showing table only. "
                        "Add `plotly` to requirements.txt for charts."
                    )

            # ── Sortable table ────────────────────────────────────────
            st.markdown("##### Alteration Frequency Table")

            # Colour pct_any column: ≥20% red, 5-20% amber, <5% green
            def _colour_pct(val):
                try:
                    v = float(val)
                    if v >= 20:  return "background-color:#FCE4D6;color:#843C0C"
                    if v >= 5:   return "background-color:#FFF2CC;color:#7F6000"
                    return "background-color:#E2EFDA;color:#375623"
                except Exception:
                    return ""

            pct_cols = [c for c in display_df.columns if c.startswith("pct_")]
            tbl_styled = display_df.style.applymap(_colour_pct, subset=pct_cols)
            st.dataframe(tbl_styled, use_container_width=True, hide_index=True)

    # ── 3. Code Q&A chat ───────────────────────────────────────────────
    if st.session_state["gene_session_id"]:
        st.divider()
        st.markdown("### 💬  Ask a Question About the Data")
        st.caption(
            "The tool writes and runs Python code against your dataset to answer "
            "your question. Examples:\n"
            "- *Which genes are mutated in >10% of samples?*\n"
            "- *What is the median VAF for KIT mutations?*\n"
            "- *How many samples have both a KIT mutation and a chromosome 14 deletion?*\n"
            "- *Show the top 5 co-occurring gene pairs.*"
        )

        with st.expander("⚙️  Code Q&A Options"):
            qa_model = st.selectbox(
                "LLM model",
                ["openai/gpt-4o", "openai/gpt-4-turbo",
                 "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"],
                key="qa_model_select",
            )
            qa_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05,
                                key="qa_temp_slider")

        # Render previous Q&A history
        for entry in st.session_state["code_qa_history"]:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                _render_qa_result(entry["result"])

        # New question input
        user_q = st.chat_input("Ask a question about the alteration data…")

        if user_q:
            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Generating and running code…"):
                    try:
                        qa_resp = requests.post(
                            f"{API_URL}/code_query/",
                            data={
                                "session_id": st.session_state["gene_session_id"],
                                "question":   user_q,
                                "llm_model":  qa_model,
                                "temperature": str(qa_temp),
                            },
                            timeout=TIMEOUT_NORMAL,
                        )
                    except requests.exceptions.Timeout:
                        st.error(
                            "⏱️ Request timed out after 5 min. "
                            "Try a simpler question or reload the data file."
                        )
                        st.stop()
                    except requests.exceptions.ConnectionError as e:
                        st.error(f"🔌 Connection error: {e}")
                        st.stop()

                if qa_resp.status_code == 200:
                    qa_result = qa_resp.json()
                    _render_qa_result(qa_result)
                    st.session_state["code_qa_history"].append({
                        "question": user_q,
                        "result":   qa_result,
                    })
                else:
                    st.error(f"Error ({qa_resp.status_code}): {qa_resp.text[:300]}")

