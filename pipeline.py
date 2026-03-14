"""
pipeline.py (top-level shortcut)
---------------------------------
Run the entire HPSM pipeline with one command:

    python pipeline.py

This is a convenience wrapper around backend.pipeline.
"""

import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backend.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
