import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import tempfile
import urllib.parse
from typing import List, Optional

from system_prompt_config import load_system_prompt
from cbioportal_curator import curate
from gene_alteration_analyst import (
    load_alteration_data,
    compute_frequencies,
    answer_question,
)

app = FastAPI(
    title="Synopsis — Literature Retrieval, cBioPortal Curation & Gene Alteration Analysis",
    description=(
        "Upload papers, supplementary files, and genomic data tables to generate "
        "evidence summaries, cBioPortal curation reports, and gene alteration analyses."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# In-memory caches (keyed by filename)
# ─────────────────────────────────────────────────────────────

_REPORT_CACHE: dict[str, str] = {}          # report filename → abs path
_ALTERATION_CACHE: dict[str, object] = {}   # session_id → {data, freq}
_FORMATTED_CACHE: dict[str, str] = {}       # zip filename → abs path


# ═════════════════════════════════════════════════════════════
# TAG: Literature Retrieval  (unchanged)
# ═════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "service": "Synopsis backend", "version": "3.0.0"}

@app.post(
    "/summarize/",
    summary="Summarise an uploaded file using RAG + LLM",
    tags=["Literature Retrieval"],
)
async def summarize(
    input_file: UploadFile = File(...),
    prompt_file: UploadFile = File(None),
    temperature: float = Form(0.7),
    top_k: int = Form(5),
):
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, input_file.filename)
        with open(input_path, "wb") as f:
            f.write(await input_file.read())

        prompt = None
        if prompt_file:
            prompt_path = os.path.join(temp_dir, prompt_file.filename)
            with open(prompt_path, "wb") as f:
                f.write(await prompt_file.read())
            prompt = load_system_prompt(prompt_path)

        try:
            from pdf_ingest import process_pdf
            from vector_store import add_embeddings, search_vector_store
            from utils import load_chat_model
            from langchain_core.messages import HumanMessage, SystemMessage

            # Ingest and embed the document
            chunks = process_pdf(input_path)
            add_embeddings(chunks)

            # Retrieve relevant context
            query_text = prompt or "Summarise the key findings of this document."
            relevant_docs = search_vector_store(query_text, k=top_k)
            context = "\n\n".join(d.page_content for d in relevant_docs)

            # Build prompt
            system = prompt or "You are a helpful assistant. Summarise the uploaded document concisely."
            user_msg = f"Context from document:\n{context}\n\nPlease summarise the key points."

            model = load_chat_model("openai/gpt-4o")
            response = model.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
            summary = response.content
            return JSONResponse(content={"summary": summary})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})


# ═════════════════════════════════════════════════════════════
# TAG: Literature Retrieval — vector store routes
# ═════════════════════════════════════════════════════════════

@app.post(
    "/ingest_pdf/",
    summary="Ingest a PDF into the vector store",
    tags=["Literature Retrieval"],
)
async def ingest_pdf(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        try:
            from pdf_ingest import process_pdf
            from vector_store import add_embeddings
            chunks = process_pdf(path)
            n = add_embeddings(chunks)
            return JSONResponse(content={"status": "ok", "chunks_added": n})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/clear_vector_store/",
    summary="Clear all documents from the vector store",
    tags=["Literature Retrieval"],
)
async def clear_vs():
    try:
        from vector_store import clear_vector_store
        clear_vector_store()
        return JSONResponse(content={"status": "cleared"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/generate_evidence/",
    summary="Answer a question using the vector store RAG pipeline",
    tags=["Literature Retrieval"],
)
async def generate_evidence(question: str = Form(...)):
    try:
        from vector_store import search_vector_store
        from utils import load_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        docs = search_vector_store(question, k=5)
        context = "\n\n".join(docs)
        system = (
            "You are an expert biomedical research assistant. "
            "Answer the question using only the provided context. "
            "Be concise and cite specific findings where possible."
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"
        model = load_chat_model("openai/gpt-4o")
        response = model.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
        return JSONResponse(content={"answer": response.content})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════
# TAG: cBioPortal Curation  (unchanged)
# ═════════════════════════════════════════════════════════════

@app.post(
    "/curate_cbioportal/",
    summary="Generate a cBioPortal curation report from a paper PDF + supplementary Excel files",
    tags=["cBioPortal Curation"],
)
async def curate_cbioportal(
    paper_pdf: UploadFile = File(..., description="Main paper PDF"),
    supplementary_files: List[UploadFile] = File(
        default=[], description="Supplementary data files (.xlsx, .xls, .csv, .tsv, .txt, .maf, .doc, .docx)"
    ),
    llm_model: str = Form(default="openai/gpt-4o"),
    temperature: float = Form(default=0.2),
):
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, paper_pdf.filename)
        with open(pdf_path, "wb") as f:
            f.write(await paper_pdf.read())

        supp_paths: list[str] = []
        for sf in supplementary_files:
            if sf.filename:
                sp = os.path.join(tmp, sf.filename)
                with open(sp, "wb") as f:
                    f.write(await sf.read())
                supp_paths.append(sp)

        try:
            report_fd, report_path = tempfile.mkstemp(
                suffix=".docx", prefix="cbio_report_"
            )
            os.close(report_fd)
            result = curate(
                pdf_path=pdf_path,
                supp_paths=supp_paths,
                llm_model=llm_model,
                temperature=temperature,
                output_path=report_path,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    encoded = urllib.parse.quote(os.path.basename(report_path))
    result["summary"]["report_download_url"] = f"/download_report/{encoded}"
    result["summary"]["report_filename"] = os.path.basename(report_path)
    _REPORT_CACHE[os.path.basename(report_path)] = result["report_path"]
    return JSONResponse(content=result["summary"])


@app.get(
    "/download_report/{filename}",
    summary="Download a previously generated curation report",
    tags=["cBioPortal Curation"],
)
async def download_report(filename: str):
    decoded = urllib.parse.unquote(filename)
    path = _REPORT_CACHE.get(decoded)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found or expired.")
    return FileResponse(
        path=path,
        filename=decoded,
        media_type=(
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        ),
    )


# ═════════════════════════════════════════════════════════════
# TAG: Gene Alteration Analysis  (new)
# ═════════════════════════════════════════════════════════════

@app.post(
    "/gene_alterations/",
    summary=(
        "Upload a MAF / Excel / CSV genomic data file and compute "
        "per-gene alteration frequencies across all samples"
    ),
    tags=["Gene Alteration Analysis"],
)
async def gene_alterations(
    data_file: UploadFile = File(
        ...,
        description=(
            "Genomic data file: MAF (.maf/.txt/.tsv), "
            "Excel (.xlsx), or CSV (.csv). "
            "Supports mutation, CNA matrix, and SV/fusion formats."
        ),
    ),
):
    """
    Parse the uploaded file and return:

    - **frequencies**: per-gene table with columns
      `n_mutated`, `pct_mutated`, `n_amp`, `pct_amp`, `n_del`, `pct_del`,
      `n_sv`, `pct_sv`, `n_any`, `pct_any`, `total_samples`
    - **summary**: top-line counts (n_samples, n_genes, data types detected)
    - **session_id**: pass back to `/code_query/` to ask follow-up questions

    Genes are sorted by `pct_any` (highest overall alteration frequency first).
    """
    with tempfile.TemporaryDirectory() as tmp:
        fpath = os.path.join(tmp, data_file.filename)
        with open(fpath, "wb") as f:
            f.write(await data_file.read())

        try:
            altdata = load_alteration_data(fpath)
            freq_df = compute_frequencies(altdata)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    # Cache for follow-up code queries
    import hashlib, time
    session_id = hashlib.md5(
        f"{data_file.filename}{time.time()}".encode()
    ).hexdigest()[:12]
    _ALTERATION_CACHE[session_id] = {"data": altdata, "freq": freq_df}

    freq_records = (
        freq_df.reset_index().to_dict(orient="records")
        if not freq_df.empty else []
    )

    summary = {
        "n_samples":     altdata.n_samples,
        "n_genes":       len(freq_df),
        "has_mutations": altdata.has_mutations,
        "has_cna":       altdata.has_cna,
        "has_sv":        altdata.has_sv,
        "top_genes":     freq_records[:10],
    }

    return JSONResponse(content={
        "session_id":  session_id,
        "summary":     summary,
        "frequencies": freq_records,
    })


@app.post(
    "/code_query/",
    summary=(
        "Ask a natural-language question about an already-loaded alteration dataset. "
        "The LLM writes and executes Python code to answer the question."
    ),
    tags=["Gene Alteration Analysis"],
)
async def code_query(
    session_id: str = Form(
        ...,
        description="session_id returned by a previous /gene_alterations/ call",
    ),
    question: str = Form(
        ...,
        description="Natural-language question about the data",
    ),
    llm_model: str = Form(default="openai/gpt-4o"),
    temperature: float = Form(default=0.2),
):
    """
    The LLM generates Python code that runs against the cached dataset
    (df_mut, df_cna, df_sv, df_freq, n_samples) and returns:

    - **answer**: the computed result (table / text / list / dict)
    - **code**: the Python code that was generated and executed
    - **explanation**: the LLM's plain-English interpretation
    - **result_type**: "dataframe" | "text" | "dict" | "list"
    - **error**: execution error traceback (null if successful)
    """
    cached = _ALTERATION_CACHE.get(session_id)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{session_id}' not found. "
                "Upload a data file via /gene_alterations/ first."
            ),
        )

    altdata = cached["data"]
    freq_df = cached["freq"]

    try:
        result = answer_question(
            data=altdata,
            freq_df=freq_df,
            question=question,
            model=llm_model,
            temperature=temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)


# ═════════════════════════════════════════════════════════════
# TAG: cBioPortal Formatting Pipeline
# ═════════════════════════════════════════════════════════════

@app.post(
    "/format_for_cbio/",
    summary=(
        "Upload a TSV/CSV/Excel file with arbitrary columns and receive "
        "a cBioPortal-ready data file + matching meta file."
    ),
    tags=["cBioPortal Formatting"],
)
async def format_for_cbio(
    data_file: UploadFile = File(
        ...,
        description=(
            "Supplemental data file with non-standard column names "
            "(.tsv, .csv, .xlsx). The pipeline auto-detects the cBioPortal "
            "format type and transforms it accordingly."
        ),
    ),
    study_id: str = Form(
        default="my_study_2025",
        description="cBioPortal cancer study identifier (e.g. 'brca_tcga_2024')",
    ),
    cbio_type: Optional[str] = Form(
        default=None,
        description=(
            "Override auto-detection. One of: clinical_patient, clinical_sample, "
            "mutation, cna_discrete, expression, structural_variant, timeline, methylation"
        ),
    ),
    curator_notes: str = Form(
        default="",
        description=(
            "Optional free-text hints for the LLM "
            "(e.g. 'survival time is in days, convert to months')"
        ),
    ),
    llm_model: str = Form(
        default="openai/gpt-4o",
        description="LLM used for transformation (e.g. 'openai/gpt-4o')",
    ),
):
    """
    Full formatting pipeline:

    1. **Parse** — auto-detect encoding + delimiter, load into DataFrame
    2. **Detect** — heuristic + LLM few-shot classification of the cBioPortal format type
    3. **Transform** — LLM rewrites columns, values, and header rows to match the spec
    4. **Package** — zip data file + meta file ready for cBioPortal upload

    Returns detection metadata, a preview of the formatted output, and a `download_url`
    for the ZIP archive containing both files.
    """
    file_bytes = await data_file.read()

    # 1. Parse
    try:
        from file_parser import parse_file
        df = parse_file(file_bytes, data_file.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Parsed file is empty.")

    # 2. Fuzzy-normalise column names (cross-spec, before detection)
    from spec_match import classify_sheet, fuzzy_normalize_columns
    original_columns = list(df.columns)
    df, column_mappings = fuzzy_normalize_columns(df)
    unmapped_columns = [c for c in original_columns if c not in column_mappings]


    # 3. Detect format type (or use caller-supplied override)
    VALID_CBIO_TYPES = {
        "clinical_patient", "clinical_sample", "mutation", "cna_discrete",
        "expression", "structural_variant", "timeline", "methylation",
    }
    if cbio_type and cbio_type not in VALID_CBIO_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid cbio_type '{cbio_type}'. "
                f"Valid values: {sorted(VALID_CBIO_TYPES)}"
            ),
        )

    if cbio_type:
        cls = None
        resolved_type = cbio_type
        detection = {
            "cbio_type":        cbio_type,
            "confidence":       1.0,
            "method":           "user_override",
            "reasoning":        "Format type manually specified by caller.",
            "column_mappings":  column_mappings,
            "unmapped_columns": unmapped_columns,
            "required_missing": [],
            "low_confidence":   False,
        }
    else:
        cls = classify_sheet(df)
        if cls.format_key == "NOT_LOADABLE":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Could not determine the cBioPortal format type "
                    f"(best candidate: {cls.all_scores[0]['format']} "
                    f"@ {cls.all_scores[0]['confidence']:.0f}%). "
                    "Please specify 'cbio_type' manually."
                ),
            )
        resolved_type = cls.format_key
        detection = {
            "cbio_type":        cls.format_key,
            "confidence":       cls.confidence,
            "method":           f"spec_match ({cls.spec_source})",
            "reasoning":        cls.verdict,
            "column_mappings":  column_mappings,
            "unmapped_columns": unmapped_columns,
            "required_missing": cls.required_missing,
            "low_confidence":   cls.confidence < 70,
        }

    # 4. Transform
    from cbio_transformer import transform_to_cbio

    try:
        result = transform_to_cbio(
            df=df,
            cbio_type=resolved_type,
            study_id=study_id,
            column_mappings=detection.get("column_mappings") or {},
            curator_notes=curator_notes,
            llm_model=llm_model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformation failed: {e}")

    # 5. Package as ZIP
    import zipfile

    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cbio_fmt_")
    os.close(zip_fd)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(result["data_filename"], result["data_content"])
        zf.writestr(result["meta_filename"], result["meta_content"])

    zip_name = os.path.basename(zip_path)
    _FORMATTED_CACHE[zip_name] = zip_path

    encoded = urllib.parse.quote(zip_name)
    preview = "\n".join(result["data_content"].splitlines()[:5])

    return JSONResponse(content={
        "detection": {
            "cbio_type":        detection["cbio_type"],
            "confidence":       detection["confidence"],
            "method":           detection["method"],
            "reasoning":        detection.get("reasoning", ""),
            "column_mappings":  detection.get("column_mappings", {}),
            "unmapped_columns": detection.get("unmapped_columns", []),
            "low_confidence":   detection.get("low_confidence", False),
        },
        "output": {
            "data_filename": result["data_filename"],
            "meta_filename": result["meta_filename"],
            "data_preview":  preview,
            "meta_content":  result["meta_content"],
        },
        "download_url": f"/download_formatted/{encoded}",
        "zip_filename":  zip_name,
    })


@app.get(
    "/download_formatted/{filename}",
    summary="Download the ZIP archive with the cBioPortal-ready data + meta files",
    tags=["cBioPortal Formatting"],
)
async def download_formatted(filename: str):
    decoded = urllib.parse.unquote(filename)
    path = _FORMATTED_CACHE.get(decoded)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Formatted archive not found or expired.")
    return FileResponse(
        path=path,
        filename=decoded,
        media_type="application/zip",
    )
