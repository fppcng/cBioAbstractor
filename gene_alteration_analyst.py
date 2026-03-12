"""
gene_alteration_analyst.py
──────────────────────────
Compute gene alteration frequencies from genomic data files and answer
natural-language questions about the data by generating and executing Python
code via an LLM.

Supported input formats
───────────────────────
  MAF (.maf, .txt, .tsv, .csv)
    Required columns: Hugo_Symbol, Tumor_Sample_Barcode
    Optional:         Variant_Classification, Variant_Type, t_AF

  Excel supplementary (.xlsx)
    Auto-detects mutation, CNA, or SV sheets using column heuristics.

  Plain CSV / TSV

Alteration types computed
─────────────────────────
  mutation   — somatic SNVs / indels from MAF-like data
  cna_amp    — high-level amplification (GISTIC ≥ 2 or log2 ≥ 1.0)
  cna_del    — deep deletion (GISTIC ≤ -2 or log2 ≤ -1.0)
  fusion/sv  — structural variants / gene fusions
  any        — sample altered by ≥1 of the above (combined)

Public API
──────────
  load_alteration_data(path)  → AlterationData
  compute_frequencies(data)   → pd.DataFrame   (gene × alteration-type)
  answer_question(data, question, model, temperature) → dict
"""

from __future__ import annotations

import io
import re
import sys
import traceback
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from utils import load_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# MAF Variant_Classification values considered non-silent / driver
DRIVER_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site",
    "De_novo_Start_InFrame", "De_novo_Start_OutOfFrame",
}

# ANNOVAR function strings that map to driver events
ANNOVAR_DRIVER = {
    "nonsynonymous snv", "stopgain", "stoploss",
    "frameshift deletion", "frameshift insertion",
    "nonframeshift deletion", "nonframeshift insertion", "splicing",
}

# Column name aliases (normalised → canonical)
COL_ALIASES: dict[str, list[str]] = {
    "hugo_symbol":            ["hugo_symbol", "gene", "gene_symbol", "gene symbol",
                               "hgnc_symbol", "symbol"],
    "tumor_sample_barcode":   ["tumor_sample_barcode", "sample_id", "sample id",
                               "tumor_barcode", "barcode", "sampleid"],
    "variant_classification": ["variant_classification", "mutation_type",
                               "mutation type", "variant_type", "exonicfunc",
                               "exonic_func", "exonicfunc.refgene",
                               "variant classification"],
    "t_af":                   ["t_af", "af", "vaf", "tumor_vaf",
                               "variant_allele_frequency",
                               "variant allele frequency mean"],
}

# ─────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────

@dataclass
class AlterationData:
    """Holds parsed genomic alteration tables."""
    mutations: pd.DataFrame = field(default_factory=pd.DataFrame)   # gene × sample MAF rows
    cna_discrete: pd.DataFrame = field(default_factory=pd.DataFrame) # gene × sample integer matrix
    cna_continuous: pd.DataFrame = field(default_factory=pd.DataFrame)
    sv: pd.DataFrame = field(default_factory=pd.DataFrame)           # SV / fusion rows
    source_file: str = ""
    n_samples: int = 0
    sample_ids: list[str] = field(default_factory=list)

    @property
    def has_mutations(self): return not self.mutations.empty
    @property
    def has_cna(self): return not self.cna_discrete.empty or not self.cna_continuous.empty
    @property
    def has_sv(self): return not self.sv.empty


# ─────────────────────────────────────────────────────────────
# Column normalisation helpers
# ─────────────────────────────────────────────────────────────

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using COL_ALIASES."""
    rename_map: dict[str, str] = {}
    for canon, aliases in COL_ALIASES.items():
        for col in df.columns:
            if str(col).lower().strip() in aliases:
                rename_map[col] = canon
                break
    return df.rename(columns=rename_map)


def _find_col(df: pd.DataFrame, canon: str) -> str | None:
    """Return the actual column name for a canonical key, or None."""
    for col in df.columns:
        if str(col).lower().strip() in COL_ALIASES.get(canon, [canon]):
            return col
    return None


# ─────────────────────────────────────────────────────────────
# Sheet / file type detection
# ─────────────────────────────────────────────────────────────

def _detect_sheet_type(df: pd.DataFrame) -> str:
    """Return 'mutation' | 'cna_matrix' | 'sv' | 'unknown'."""
    cols_lower = {str(c).lower() for c in df.columns}
    header_str = " ".join(str(c).lower() for c in df.columns)

    # MAF-like
    if any(a in cols_lower for a in COL_ALIASES["hugo_symbol"]) and \
       any(a in cols_lower for a in COL_ALIASES["tumor_sample_barcode"]):
        return "mutation"

    # SV / fusion
    sv_signals = {"sv type", "sv_type", "left gene", "right gene",
                  "site1_hugo_symbol", "site2_hugo_symbol", "fusion", "breakpoint",
                  "left_gene", "right_gene"}
    if sv_signals & cols_lower:
        return "sv"

    # CNA matrix: first col is gene names, rest are sample IDs with numeric values
    if df.shape[1] > 3:
        non_first = df.iloc[:, 1:]
        try:
            numeric_frac = non_first.apply(pd.to_numeric, errors="coerce").notna().mean().mean()
            if numeric_frac > 0.7:
                return "cna_matrix"
        except Exception:
            pass

    return "unknown"


# ─────────────────────────────────────────────────────────────
# Readers
# ─────────────────────────────────────────────────────────────

def _read_flat_file(path: str) -> pd.DataFrame:
    """Read MAF / TSV / CSV, skipping comment lines."""
    sep = "\t" if path.endswith((".maf", ".tsv", ".txt")) else ","
    rows: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("#"):
                rows.append(line)
    return pd.read_csv(io.StringIO("".join(rows)), sep=sep, low_memory=False)


def _read_excel(path: str) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    sheets: dict[str, pd.DataFrame] = {}
    for name in xl.sheet_names:
        df = xl.parse(name)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if not df.empty:
            sheets[name] = df
    return sheets


# ─────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────

def _parse_mutation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a MAF-like DataFrame to: hugo_symbol, tumor_sample_barcode,
    variant_classification, t_af."""
    df = _normalise_cols(df.copy())
    required = ["hugo_symbol", "tumor_sample_barcode"]
    for req in required:
        if req not in df.columns:
            raise ValueError(
                f"Could not find required column '{req}'. "
                f"Available: {list(df.columns)}"
            )
    keep = ["hugo_symbol", "tumor_sample_barcode"]
    for opt in ["variant_classification", "t_af"]:
        if opt in df.columns:
            keep.append(opt)
    df = df[keep].dropna(subset=["hugo_symbol", "tumor_sample_barcode"])
    df["hugo_symbol"] = df["hugo_symbol"].astype(str).str.strip().str.upper()
    df["tumor_sample_barcode"] = df["tumor_sample_barcode"].astype(str).str.strip()

    # Normalise variant_classification if present
    if "variant_classification" in df.columns:
        vc = df["variant_classification"].astype(str).str.lower().str.strip()
        # ANNOVAR → MAF conversion
        annovar_map = {
            "nonsynonymous snv":      "Missense_Mutation",
            "synonymous snv":         "Silent",
            "stopgain":               "Nonsense_Mutation",
            "stoploss":               "Nonstop_Mutation",
            "frameshift deletion":    "Frame_Shift_Del",
            "frameshift insertion":   "Frame_Shift_Ins",
            "nonframeshift deletion": "In_Frame_Del",
            "nonframeshift insertion":"In_Frame_Ins",
            "splicing":               "Splice_Site",
        }
        df["variant_classification"] = vc.map(annovar_map).fillna(
            df["variant_classification"])

    return df


def _parse_cna_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Parse a gene × sample CNA matrix.
    Assumes first column is gene symbol and remaining columns are samples."""
    gene_col = df.columns[0]
    df = df.copy()
    df[gene_col] = df[gene_col].astype(str).str.strip().str.upper()
    df = df.set_index(gene_col)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index.name = "hugo_symbol"
    return df


def _parse_sv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise SV / fusion rows to: sample_id, gene1, gene2, sv_class."""
    df = df.copy()
    cols_lower = {str(c).lower(): c for c in df.columns}

    def pick(*candidates):
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        return None

    sample_col = pick("sample_id", "sample id", "tumor_sample_barcode", "barcode", "sampleid")
    gene1_col  = pick("left_gene", "left gene", "gene1", "site1_hugo_symbol",
                      "gene a", "genea", "partner1")
    gene2_col  = pick("right_gene", "right gene", "gene2", "site2_hugo_symbol",
                      "gene b", "geneb", "partner2")
    type_col   = pick("sv_type", "sv type", "class", "event_type", "type")

    out_rows = []
    for _, row in df.iterrows():
        sample = str(row[sample_col]).strip() if sample_col else "unknown"
        g1 = str(row[gene1_col]).strip().upper() if gene1_col else ""
        g2 = str(row[gene2_col]).strip().upper() if gene2_col else ""
        sv_class = str(row[type_col]).strip().upper() if type_col else "SV"
        for gene in {g for g in [g1, g2] if g and g not in {"", "NAN", "NONE", "-"}}:
            out_rows.append({
                "hugo_symbol":         gene,
                "tumor_sample_barcode": sample,
                "sv_class":            sv_class,
            })
    return pd.DataFrame(out_rows)


# ─────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────

def load_alteration_data(path: str) -> AlterationData:
    """
    Load a genomic data file and return an AlterationData object.

    Supports: .maf, .tsv, .csv, .txt (flat) and .xlsx (multi-sheet).
    """
    ext = Path(path).suffix.lower()
    data = AlterationData(source_file=Path(path).name)

    if ext == ".xlsx":
        sheets = _read_excel(path)
        for sheet_name, df in sheets.items():
            stype = _detect_sheet_type(df)
            if stype == "mutation":
                try:
                    data.mutations = pd.concat(
                        [data.mutations, _parse_mutation_df(df)], ignore_index=True
                    ) if not data.mutations.empty else _parse_mutation_df(df)
                except Exception:
                    pass
            elif stype == "cna_matrix":
                try:
                    parsed = _parse_cna_matrix(df)
                    # Heuristic: discrete if values are small integers, else continuous
                    vals = parsed.values.flatten()
                    vals = vals[~np.isnan(vals)]
                    if len(vals) and np.all(np.abs(vals) <= 4) and np.all(vals == vals.astype(int)):
                        data.cna_discrete = parsed
                    else:
                        data.cna_continuous = parsed
                except Exception:
                    pass
            elif stype == "sv":
                try:
                    parsed = _parse_sv_df(df)
                    data.sv = pd.concat(
                        [data.sv, parsed], ignore_index=True
                    ) if not data.sv.empty else parsed
                except Exception:
                    pass

    else:
        df = _read_flat_file(path)
        stype = _detect_sheet_type(df)
        if stype == "mutation":
            data.mutations = _parse_mutation_df(df)
        elif stype == "cna_matrix":
            parsed = _parse_cna_matrix(df)
            vals = parsed.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) and np.all(np.abs(vals) <= 4) and np.all(vals == vals.astype(int)):
                data.cna_discrete = parsed
            else:
                data.cna_continuous = parsed
        elif stype == "sv":
            data.sv = _parse_sv_df(df)
        else:
            # Last resort: try parsing as mutation
            try:
                data.mutations = _parse_mutation_df(df)
            except Exception:
                pass

    # Collect all sample IDs
    sample_sets: list[set[str]] = []
    if data.has_mutations:
        sample_sets.append(set(data.mutations["tumor_sample_barcode"].unique()))
    if not data.cna_discrete.empty:
        sample_sets.append(set(data.cna_discrete.columns))
    if not data.cna_continuous.empty:
        sample_sets.append(set(data.cna_continuous.columns))
    if data.has_sv:
        sample_sets.append(set(data.sv["tumor_sample_barcode"].unique()))

    all_samples = set().union(*sample_sets) if sample_sets else set()
    data.sample_ids = sorted(all_samples)
    data.n_samples = len(all_samples)

    return data


# ─────────────────────────────────────────────────────────────
# Frequency computation
# ─────────────────────────────────────────────────────────────

def compute_frequencies(data: AlterationData) -> pd.DataFrame:
    """
    Return a DataFrame indexed by gene with columns:
      n_mutated, pct_mutated,
      n_amp, pct_amp,
      n_del, pct_del,
      n_sv, pct_sv,
      n_any, pct_any,
      total_samples
    """
    n = data.n_samples
    if n == 0:
        return pd.DataFrame()

    genes: set[str] = set()
    mut_by_gene: dict[str, set[str]] = {}
    amp_by_gene: dict[str, set[str]] = {}
    del_by_gene: dict[str, set[str]] = {}
    sv_by_gene:  dict[str, set[str]] = {}

    # ── Mutations ──────────────────────────────────────────────
    if data.has_mutations:
        mdf = data.mutations.copy()
        # Filter to driver events if classification is available
        if "variant_classification" in mdf.columns:
            driver_mask = mdf["variant_classification"].isin(DRIVER_CLASSES)
            # Also catch ANNOVAR-style strings
            annovar_mask = mdf["variant_classification"].str.lower().isin(ANNOVAR_DRIVER)
            mdf = mdf[driver_mask | annovar_mask]
        for gene, grp in mdf.groupby("hugo_symbol"):
            gene = str(gene).upper()
            genes.add(gene)
            mut_by_gene[gene] = set(grp["tumor_sample_barcode"].unique())

    # ── CNA — discrete matrix ──────────────────────────────────
    if not data.cna_discrete.empty:
        cdf = data.cna_discrete
        for gene in cdf.index:
            gene = str(gene).upper()
            genes.add(gene)
            row = cdf.loc[cdf.index == gene].iloc[0]
            amp_samples = set(row[row >= 2].index.astype(str))
            del_samples = set(row[row <= -2].index.astype(str))
            if amp_samples:
                amp_by_gene[gene] = amp_by_gene.get(gene, set()) | amp_samples
            if del_samples:
                del_by_gene[gene] = del_by_gene.get(gene, set()) | del_samples

    # ── CNA — continuous (log2) ────────────────────────────────
    if not data.cna_continuous.empty:
        cdf = data.cna_continuous
        for gene in cdf.index:
            gene = str(gene).upper()
            genes.add(gene)
            row = cdf.loc[cdf.index == gene].iloc[0]
            amp_samples = set(row[row >= 1.0].index.astype(str))
            del_samples = set(row[row <= -1.0].index.astype(str))
            if amp_samples:
                amp_by_gene[gene] = amp_by_gene.get(gene, set()) | amp_samples
            if del_samples:
                del_by_gene[gene] = del_by_gene.get(gene, set()) | del_samples

    # ── SV / Fusions ───────────────────────────────────────────
    if data.has_sv:
        for gene, grp in data.sv.groupby("hugo_symbol"):
            gene = str(gene).upper()
            genes.add(gene)
            sv_by_gene[gene] = set(grp["tumor_sample_barcode"].unique())

    # ── Build result DataFrame ─────────────────────────────────
    rows = []
    for gene in sorted(genes):
        m_samp = mut_by_gene.get(gene, set())
        a_samp = amp_by_gene.get(gene, set())
        d_samp = del_by_gene.get(gene, set())
        s_samp = sv_by_gene.get(gene, set())
        any_samp = m_samp | a_samp | d_samp | s_samp
        rows.append({
            "gene":          gene,
            "n_mutated":     len(m_samp),
            "pct_mutated":   round(100 * len(m_samp) / n, 1),
            "n_amp":         len(a_samp),
            "pct_amp":       round(100 * len(a_samp) / n, 1),
            "n_del":         len(d_samp),
            "pct_del":       round(100 * len(d_samp) / n, 1),
            "n_sv":          len(s_samp),
            "pct_sv":        round(100 * len(s_samp) / n, 1),
            "n_any":         len(any_samp),
            "pct_any":       round(100 * len(any_samp) / n, 1),
            "total_samples": n,
        })

    df = pd.DataFrame(rows).set_index("gene")
    return df.sort_values("pct_any", ascending=False)


# ─────────────────────────────────────────────────────────────
# LLM code-interpreter Q&A
# ─────────────────────────────────────────────────────────────

_CODE_SYSTEM_PROMPT = """
You are a bioinformatics data analyst with expert knowledge of Python, pandas,
and cancer genomics.

The user has uploaded a genomic data file. You have access to the following
pre-loaded Python variables:

  df_mut     pd.DataFrame   — somatic mutation rows
                              columns: hugo_symbol, tumor_sample_barcode,
                                       variant_classification (if present), t_af (if present)

  df_cna     pd.DataFrame   — gene × sample CNA matrix (discrete GISTIC calls or log2 ratios)
                              index = gene symbol, columns = sample IDs

  df_sv      pd.DataFrame   — structural variant / fusion rows
                              columns: hugo_symbol, tumor_sample_barcode, sv_class

  df_freq    pd.DataFrame   — pre-computed alteration frequency table
                              columns: n_mutated, pct_mutated, n_amp, pct_amp,
                                       n_del, pct_del, n_sv, pct_sv,
                                       n_any, pct_any, total_samples
                              index = gene symbol, sorted by pct_any descending

  n_samples  int            — total unique sample count

Any of these DataFrames may be empty if that data type was not present in
the uploaded file.

When asked a question:
1. Write Python code using ONLY these variables (no file I/O, no imports except
   what is already available: pandas as pd, numpy as np).
2. Store the final answer in a variable called `result`. This must be either:
   - a string (plain text answer)
   - a pd.DataFrame (will be displayed as a table)
   - a dict (will be displayed as JSON)
   - a list (will be displayed as a list)
3. Wrap ALL your code in a single ```python ... ``` fenced block.
4. After the code block, provide a brief plain-English explanation of what the
   code does and what the answer means clinically / biologically.

Rules:
  - Do NOT import new modules or read files.
  - Do NOT use print() — always assign to `result`.
  - Handle empty DataFrames gracefully with `if df.empty: result = "No data available"`
  - Be precise: if asked for "top N genes", sort and use .head(N).
"""


def answer_question(
    data: AlterationData,
    freq_df: pd.DataFrame,
    question: str,
    model: str = "openai/gpt-4o",
    temperature: float = 0.2,
) -> dict[str, Any]:
    """
    Use an LLM to generate and execute Python code that answers *question*
    using the pre-loaded alteration data.

    Returns
    -------
    {
        "answer":      str | list | dict,  # the `result` variable
        "code":        str,                # the generated Python code
        "explanation": str,                # the LLM's prose explanation
        "error":       str | None,         # execution error if any
        "result_type": str,                # "dataframe" | "text" | "dict" | "list"
    }
    """
    llm = load_chat_model(model)
    messages = [
        SystemMessage(content=_CODE_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    llm_response = llm.invoke(messages)
    full_text = llm_response.content.strip()

    # Extract code block
    code_match = re.search(r"```python\s*(.*?)```", full_text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""

    # Extract explanation (text after the code block)
    explanation = re.sub(r"```python.*?```", "", full_text, flags=re.DOTALL).strip()

    if not code:
        return {
            "answer":      full_text,
            "code":        "",
            "explanation": full_text,
            "error":       None,
            "result_type": "text",
        }

    # Build execution namespace with the pre-loaded data
    exec_globals: dict[str, Any] = {
        "pd":        pd,
        "np":        np,
        "df_mut":    data.mutations.copy() if data.has_mutations else pd.DataFrame(),
        "df_cna":    (data.cna_discrete.copy() if not data.cna_discrete.empty
                      else data.cna_continuous.copy()),
        "df_sv":     data.sv.copy() if data.has_sv else pd.DataFrame(),
        "df_freq":   freq_df.copy() if not freq_df.empty else pd.DataFrame(),
        "n_samples": data.n_samples,
        "result":    None,
    }

    error = None
    try:
        exec(code, exec_globals)  # noqa: S102
    except Exception:
        error = traceback.format_exc()

    raw_result = exec_globals.get("result")

    # Determine result type and serialise
    if isinstance(raw_result, pd.DataFrame):
        result_type = "dataframe"
        answer = raw_result.reset_index().to_dict(orient="records")
    elif isinstance(raw_result, (pd.Series,)):
        result_type = "dataframe"
        answer = raw_result.reset_index().to_dict(orient="records")
    elif isinstance(raw_result, dict):
        result_type = "dict"
        answer = {str(k): v for k, v in raw_result.items()}
    elif isinstance(raw_result, (list, tuple)):
        result_type = "list"
        answer = list(raw_result)
    elif raw_result is None and error:
        result_type = "text"
        answer = f"Code execution failed:\n{error}"
    else:
        result_type = "text"
        answer = str(raw_result) if raw_result is not None else "No result produced."

    return {
        "answer":      answer,
        "code":        code,
        "explanation": explanation,
        "error":       error,
        "result_type": result_type,
    }
