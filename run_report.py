#!/usr/bin/env python3
"""
run_report.py
─────────────
Standalone driver for the Synopsis cBioPortal curation report generator.

Usage
-----
    python run_report.py  <paper.pdf>  <supp1.xlsx> [supp2.xlsx ...]  [options]

Options
-------
    --model    openai/gpt-4o          LLM model for metadata extraction
    --out      ./my_report.docx       output path (auto-generated if omitted)
    --temp     0.2                    LLM temperature

Examples
--------
    # Basic usage
    python run_report.py paper.pdf Supplementary_Data_1.xlsx Supplementary_Data_3.xlsx

    # All supplementary files in a directory
    python run_report.py paper.pdf supp/*.xlsx

    # Custom output path
    python run_report.py paper.pdf supp/*.xlsx --out reports/curation_report.docx

    # Use a different model
    python run_report.py paper.pdf supp/*.xlsx --model bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0

Requirements
------------
    pip install python-docx PyPDF2 pandas openpyxl langchain-openai langchain-core
    export OPENAI_API_KEY="sk-..."

Environment variables
---------------------
    OPENAI_API_KEY    required for openai/* models
    AWS_PROFILE       required for bedrock/* models
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a cBioPortal curation report from a paper PDF "
                    "and supplementary Excel files."
    )
    parser.add_argument("pdf",  help="Path to the main paper PDF")
    parser.add_argument("supp", nargs="+", help="Path(s) to supplementary Excel file(s)")
    parser.add_argument("--model", default="openai/gpt-4o",
                        help="LLM model string (default: openai/gpt-4o)")
    parser.add_argument("--out",   default=None,
                        help="Output .docx path (auto-generated if omitted)")
    parser.add_argument("--temp",  type=float, default=0.2,
                        help="LLM temperature (default: 0.2)")
    args = parser.parse_args()

    # ── validate inputs ───────────────────────────────────────────────────
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    supp_paths: list[str] = []
    for p in args.supp:
        sp = Path(p)
        if not sp.exists():
            print(f"WARNING: supplementary file not found — skipping: {sp}",
                  file=sys.stderr)
        else:
            supp_paths.append(str(sp))

    if not supp_paths:
        print("ERROR: No supplementary files found.", file=sys.stderr)
        sys.exit(1)

    # ── run curation ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Synopsis — cBioPortal Curation Report Generator")
    print(f"{'='*60}")
    print(f"  PDF:   {pdf_path}")
    print(f"  Supp:  {len(supp_paths)} file(s)")
    print(f"  Model: {args.model}")
    print(f"{'='*60}\n")

    # Import here so missing deps give a clear message
    try:
        from cbioportal_curator import curate
    except ImportError as e:
        print(f"ERROR: Cannot import cbioportal_curator: {e}", file=sys.stderr)
        print("Make sure you are running from the bundle directory.", file=sys.stderr)
        sys.exit(1)

    print("Step 1/3  Extracting metadata from PDF via LLM…")
    try:
        result = curate(
            pdf_path   = str(pdf_path),
            supp_paths = supp_paths,
            llm_model  = args.model,
            temperature= args.temp,
            output_path= args.out,
        )
    except Exception as e:
        print(f"\nERROR during curation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── print summary ─────────────────────────────────────────────────────
    report_path = result["report_path"]
    summary     = result["summary"]

    print("\nStep 2/3  Analysing supplementary files…  done")
    print("Step 3/3  Building report…                done")
    print(f"\n{'='*60}")
    print(f"  REPORT SAVED:  {report_path}")
    print(f"{'='*60}")
    print(f"  Study ID:       {summary.get('study_id','')}")
    print(f"  Cancer type:    {summary.get('cancer_type','')}")
    print(f"  Samples:        {summary.get('num_samples','?')}")
    print(f"  Ref genome:     {summary.get('reference_genome','')}")
    print(f"  Files analysed: {summary.get('files_analysed',0)}")
    print(f"  Sheets:         {summary.get('sheets_analysed',0)}")
    print(f"  HIGH priority:  {summary.get('high_priority',0)}")
    print(f"  MEDIUM priority:{summary.get('medium_priority',0)}")
    print(f"  Not loadable:   {summary.get('not_loadable',0)}")
    print()
    print("  Per-file breakdown:")
    for fb in summary.get("file_breakdown", []):
        conf = fb.get("confidence", 0)
        print(f"    {fb['file']:35s}  {fb['cbio_format']:30s}"
              f"  {fb['curability']:7s}  conf={conf:.0f}%")
    print()

    return report_path


if __name__ == "__main__":
    main()
