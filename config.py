"""
config.py
---------
Central configuration for the Synopsis backend.
All tunable constants live here.
"""
import os

# ── API keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")   # optional, used by cbio_detector LLM path

# ── Paths ─────────────────────────────────────────────────────────────────────
VECTOR_STORE_PATH  = os.getenv("VECTOR_STORE_PATH", "./vector_store")
PUBLICATIONS_DIR   = os.getenv("PUBLICATIONS_DIR",  "./publications")
FEW_SHOT_DIR       = os.getenv("FEW_SHOT_DIR",      "./few_shot_examples")

# ── Detection settings ────────────────────────────────────────────────────────
DETECTION_SAMPLE_ROWS          = 10    # rows sampled for type detection
TRANSFORM_SAMPLE_ROWS          = 20    # rows sampled for LLM transform
DETECTION_CONFIDENCE_THRESHOLD = 0.6  # below this → fall back to LLM detection

# ── cBioPortal format IDs ─────────────────────────────────────────────────────
CBIO_FORMAT_IDS = [
    "clinical_patient",
    "clinical_sample",
    "mutation",
    "cna_discrete",
    "expression",
    "structural_variant",
    "timeline",
    "methylation",
]
