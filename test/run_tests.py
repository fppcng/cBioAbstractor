#!/usr/bin/env python3
"""
run_tests.py
────────────
End-to-end test runner for the cBioPortal formatting pipeline.

For each NNN.input.txt in test_set/input/ that has a matching
NNN.output.txt in test_set/output/, runs the full pipeline:

    parse → fuzzy-normalise → classify → [LLM transform]

and compares the result against the expected output.

Metrics per test:
  • Format type detection + confidence (vs. expected)
  • Column mapping: precision / recall / F1 vs expected output columns
  • Required columns present / missing
  • Data row count match
  • Header rows check (clinical files)
  • Cell-level exact match rate (LLM mode only)

Usage:
  python run_tests.py                       # full pipeline (LLM enabled)
  python run_tests.py --no-llm             # classification + mapping only
  python run_tests.py --id 007 008         # run specific test(s)
  python run_tests.py --model openai/gpt-4-turbo
  python run_tests.py --study-id brca_test_2024
"""

import sys
import io
import argparse
import textwrap
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

W = 72  # display width

# ─────────────────────────────────────────────────────────────────────────────
# Parse expected output files
# ─────────────────────────────────────────────────────────────────────────────

def _parse_expected(path: Path) -> dict:
    """
    Parse a cBioPortal-formatted expected output file into its components.

    Returns a dict with:
      text          – raw file text
      header_lines  – list of '#…' lines (empty for non-clinical formats)
      col_row       – list of column names (line 5 for clinical, line 1 otherwise)
      data_df       – DataFrame of data rows (no header rows)
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]

    header_lines = [l for l in lines if l.startswith("#")]
    data_lines   = [l for l in lines if not l.startswith("#")]

    if not data_lines:
        return dict(text=text, header_lines=header_lines,
                    col_row=[], data_df=pd.DataFrame())

    col_row = data_lines[0].split("\t")
    try:
        data_df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t", dtype=str)
    except Exception:
        data_df = pd.DataFrame()

    return dict(text=text, header_lines=header_lines, col_row=col_row, data_df=data_df)


def _expected_cbio_type(parsed: dict) -> str:
    """Infer cbio_type from the expected output file structure."""
    upper = {c.upper() for c in parsed["col_row"]}

    if parsed["header_lines"]:
        if "SAMPLE_ID" in upper and "PATIENT_ID" in upper:
            return "CLINICAL_SAMPLE"
        return "CLINICAL_PATIENT"

    if "HUGO_SYMBOL" in upper:
        return "MUTATION_MAF"

    if "SAMPLE_ID" in upper and len(upper) <= 4:
        return "CLINICAL_SAMPLE"   # simplified sample list

    if any("GENE" in c for c in upper):
        return "MUTATION_MAF"

    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _col_metrics(pred_cols: list, exp_cols: list) -> dict:
    pred_set = {c.upper() for c in pred_cols}
    exp_set  = {c.upper() for c in exp_cols}
    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set  - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(
        precision=prec, recall=rec, f1=f1,
        tp=tp, fp=fp, fn=fn,
        missing=sorted(exp_set - pred_set),
        extra=sorted(pred_set - exp_set),
    )


def _cell_match_rate(pred_df: pd.DataFrame, exp_df: pd.DataFrame) -> float:
    """Exact-match fraction across shared columns, aligned by row position."""
    if pred_df is None or pred_df.empty or exp_df.empty:
        return 0.0

    exp_upper  = {c.upper(): c for c in exp_df.columns}
    pred_upper = {c.upper(): c for c in pred_df.columns}
    shared = sorted(set(exp_upper) & set(pred_upper))
    if not shared:
        return 0.0

    exp_sub  = exp_df [[exp_upper [c] for c in shared]].reset_index(drop=True)
    pred_sub = pred_df[[pred_upper[c] for c in shared]].reset_index(drop=True)
    n_rows   = min(len(exp_sub), len(pred_sub))
    if n_rows == 0:
        return 0.0

    exp_sub  = exp_sub .iloc[:n_rows].fillna("").astype(str)
    pred_sub = pred_sub.iloc[:n_rows].fillna("").astype(str)
    pred_sub.columns = exp_sub.columns

    total   = exp_sub.size
    matches = int((exp_sub == pred_sub).sum().sum())
    return matches / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

# Maps classify_sheet format keys → cbio_transformer keys
_FORMAT_KEY_MAP = {
    "CLINICAL_PATIENT":   "clinical_patient",
    "CLINICAL_SAMPLE":    "clinical_sample",
    "MUTATION_MAF":       "mutation",
    "DISCRETE_CNA":       "cna_discrete",
    "CONTINUOUS_CNA":     "cna_discrete",
    "EXPRESSION":         "expression",
    "STRUCTURAL_VARIANT": "structural_variant",
    "METHYLATION":        "methylation",
    "TIMELINE":           "timeline",
}


def run_pipeline(input_path: Path, use_llm: bool, llm_model: str, study_id: str) -> dict:
    from file_parser import parse_file
    from spec_match  import classify_sheet, fuzzy_normalize_columns

    file_bytes = input_path.read_bytes()
    df = parse_file(file_bytes, input_path.name)

    original_columns = list(df.columns)
    df, column_mappings = fuzzy_normalize_columns(df)
    unmapped = [c for c in original_columns if c not in column_mappings]

    cls = classify_sheet(df)

    result = dict(
        column_mappings=column_mappings,
        unmapped_columns=unmapped,
        renamed_columns=list(df.columns),
        classification=dict(
            format_key=cls.format_key,
            confidence=cls.confidence,
            required_present=cls.required_present,
            required_missing=cls.required_missing,
        ),
        llm_output=None,
        llm_df=None,
        llm_header_lines=[],
        llm_error=None,
    )

    if use_llm and cls.format_key != "NOT_LOADABLE":
        cbio_key = _FORMAT_KEY_MAP.get(cls.format_key)
        if cbio_key:
            from cbio_transformer import transform_to_cbio
            try:
                out = transform_to_cbio(
                    df=df,
                    cbio_type=cbio_key,
                    study_id=study_id,
                    column_mappings={},
                    curator_notes="",
                    llm_model=llm_model,
                )
                llm_text = out["data_content"]
                result["llm_output"] = llm_text

                lines = [l for l in llm_text.splitlines() if l.strip()]
                result["llm_header_lines"] = [l for l in lines if l.startswith("#")]
                data_lines = [l for l in lines if not l.startswith("#")]
                if data_lines:
                    try:
                        result["llm_df"] = pd.read_csv(
                            io.StringIO("\n".join(data_lines)), sep="\t", dtype=str
                        )
                    except Exception:
                        pass
            except Exception as e:
                result["llm_error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(label: str, value: float, width: int = 22) -> str:
    filled = int(round(value * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:32s} [{bar}] {value * 100:5.1f}%"


def _section(title: str):
    print(f"\n{'─' * W}")
    print(f"  {title}")
    print(f"{'─' * W}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-test report
# ─────────────────────────────────────────────────────────────────────────────

def report_test(test_id: str, pipeline: dict, expected: dict, use_llm: bool) -> dict:
    exp_type   = _expected_cbio_type(expected)
    cls        = pipeline["classification"]
    detected   = cls["format_key"]
    confidence = cls["confidence"]
    type_match = detected.upper() == exp_type.upper()

    exp_cols  = expected["col_row"]
    pred_cols = pipeline["renamed_columns"]
    map_m     = _col_metrics(pred_cols, exp_cols)

    _section(f"TEST {test_id}")

    # ── Format detection ──
    icon = "✅" if type_match else "❌"
    print(f"  Format detection:  {icon}  {detected}  (confidence {confidence:.0f}%)")
    if not type_match:
        print(f"                     expected: {exp_type}")

    # ── Column mapping ──
    n_input = len(pipeline["column_mappings"]) + len(pipeline["unmapped_columns"])
    print(f"\n  Column mapping  ({n_input} input columns → "
          f"{len(pipeline['column_mappings'])} mapped, "
          f"{len(pipeline['unmapped_columns'])} unmapped):")
    print(_bar("vs expected — Precision", map_m["precision"]))
    print(_bar("vs expected — Recall",    map_m["recall"]))
    print(_bar("vs expected — F1",        map_m["f1"]))

    if map_m["missing"]:
        s = textwrap.fill(", ".join(map_m["missing"]),
                          width=W - 22, subsequent_indent=" " * 22)
        print(f"    ⚠ Missing:   {s}")
    if map_m["extra"]:
        s = textwrap.fill(", ".join(map_m["extra"]),
                          width=W - 22, subsequent_indent=" " * 22)
        print(f"    + Extra:     {s}")

    # ── Required columns ──
    if cls["required_missing"]:
        print(f"\n  ⚠ Required columns missing: {', '.join(cls['required_missing'])}")
    else:
        print(f"\n  ✅ All required columns present")

    # ── LLM output quality ──
    llm_col_f1  = None
    cell_rate   = None

    if use_llm:
        print()
        if pipeline["llm_error"]:
            print(f"  ❌ LLM error: {pipeline['llm_error'][:100]}")
        else:
            llm_df  = pipeline.get("llm_df")
            exp_df  = expected["data_df"]

            # Row count
            pred_rows = len(llm_df) if llm_df is not None else 0
            exp_rows  = len(exp_df)
            row_icon  = "✅" if pred_rows == exp_rows else "⚠ "
            print(f"  {row_icon} Data rows:   predicted {pred_rows}  /  expected {exp_rows}")

            # Header rows (clinical)
            n_exp_hdr  = len(expected["header_lines"])
            n_pred_hdr = len(pipeline["llm_header_lines"])
            if n_exp_hdr > 0:
                hdr_icon = "✅" if n_pred_hdr == n_exp_hdr else "❌"
                print(f"  {hdr_icon} Header rows: predicted {n_pred_hdr}  /  expected {n_exp_hdr}")

            # Column quality
            llm_cols = list(llm_df.columns) if llm_df is not None else []
            llm_m    = _col_metrics(llm_cols, exp_cols)
            llm_col_f1 = llm_m["f1"]
            print(_bar("LLM columns — Precision", llm_m["precision"]))
            print(_bar("LLM columns — Recall",    llm_m["recall"]))
            print(_bar("LLM columns — F1",        llm_m["f1"]))
            if llm_m["missing"]:
                s = textwrap.fill(", ".join(llm_m["missing"]),
                                  width=W - 22, subsequent_indent=" " * 22)
                print(f"    ⚠ Missing:   {s}")

            # Cell match
            cell_rate = _cell_match_rate(llm_df, exp_df)
            print(_bar("Cell exact match",         cell_rate))

    return dict(
        test_id=test_id,
        type_match=type_match,
        confidence=confidence,
        map_f1=map_m["f1"],
        llm_col_f1=llm_col_f1,
        cell_rate=cell_rate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list, use_llm: bool):
    print(f"\n{'═' * W}")
    print(f"  SUMMARY  ({len(results)} tests)")
    print(f"{'═' * W}")

    n_type_ok  = sum(1 for r in results if r["type_match"])
    avg_conf   = sum(r["confidence"] for r in results) / len(results)
    avg_map_f1 = sum(r["map_f1"]     for r in results) / len(results)

    print(f"\n  Format detection accuracy:  {n_type_ok} / {len(results)}")
    print(_bar("Avg detection confidence",    avg_conf / 100))
    print(_bar("Avg column mapping F1",       avg_map_f1))

    if use_llm:
        llm_res = [r for r in results if r["llm_col_f1"] is not None]
        if llm_res:
            avg_llm_f1 = sum(r["llm_col_f1"] for r in llm_res) / len(llm_res)
            cell_res   = [r["cell_rate"] for r in llm_res if r["cell_rate"] is not None]
            avg_cell   = sum(cell_res) / len(cell_res) if cell_res else 0.0
            print(_bar("Avg LLM column F1",          avg_llm_f1))
            print(_bar("Avg cell exact match",        avg_cell))

    print()
    print(f"  {'ID':<6}  {'Type':5}  {'Conf':>5}  {'MapF1':>6}  "
          + ("{'LLM F1':>6}  {'Cell%':>6}" if use_llm else ""))
    print(f"  {'──':<6}  {'─────':5}  {'────':>5}  {'─────':>6}  "
          + ("{'──────':>6}  {'─────':>6}" if use_llm else ""))
    for r in results:
        row = (f"  {r['test_id']:<6}  "
               f"{'✅' if r['type_match'] else '❌':5}  "
               f"{r['confidence']:>5.0f}  "
               f"{r['map_f1']*100:>5.1f}%")
        if use_llm:
            lf1 = f"{r['llm_col_f1']*100:>5.1f}%" if r["llm_col_f1"] is not None else "   N/A"
            cr  = f"{r['cell_rate']*100:>5.1f}%"  if r["cell_rate"]  is not None else "   N/A"
            row += f"  {lf1:>7}  {cr:>7}"
        print(row)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM transform (classification + column mapping only)",
    )
    parser.add_argument(
        "--id", nargs="+",
        help="Run only specific test IDs, e.g. --id 007 008",
    )
    parser.add_argument(
        "--model", default="openai/gpt-4o",
        help="LLM model for the transform step (default: openai/gpt-4o)",
    )
    parser.add_argument(
        "--study-id", default="test_study",
        help="cancer_study_identifier passed to the transformer",
    )
    args = parser.parse_args()

    use_llm = not args.no_llm

    test_dir   = Path(__file__).resolve().parent
    input_dir  = test_dir / "input"
    output_dir = test_dir / "output"

    pairs = [
        (inp, output_dir / inp.name.replace("input", "output"))
        for inp in sorted(input_dir.glob("*.input.txt"))
    ]
    pairs = [(inp, out) for inp, out in pairs if out.exists()]

    if args.id:
        pairs = [(inp, out) for inp, out in pairs
                 if any(inp.name.startswith(i) for i in args.id)]

    if not pairs:
        print("No matching test pairs found.")
        sys.exit(1)

    mode = "FULL PIPELINE (LLM enabled)" if use_llm else "CLASSIFICATION + COLUMN MAPPING ONLY"
    print(f"\n{'═' * W}")
    print(f"  cBioPortal Formatting Pipeline — Test Suite")
    print(f"  Mode:   {mode}")
    if use_llm:
        print(f"  Model:  {args.model}")
    print(f"  Tests:  {len(pairs)}")
    print(f"{'═' * W}")

    all_results = []
    for inp_path, out_path in pairs:
        test_id = inp_path.stem.split(".")[0]
        print(f"\n  Running {test_id}…", end=" ", flush=True)
        try:
            pipeline = run_pipeline(
                inp_path,
                use_llm=use_llm,
                llm_model=args.model,
                study_id=args.study_id,
            )
        except Exception as e:
            print(f"FAILED — {e}")
            continue
        print("done")

        expected = _parse_expected(out_path)
        res = report_test(test_id, pipeline, expected, use_llm=use_llm)
        all_results.append(res)

    if len(all_results) > 1:
        print_summary(all_results, use_llm=use_llm)


if __name__ == "__main__":
    main()
