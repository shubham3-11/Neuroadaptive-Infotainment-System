#!/usr/bin/env python3
"""
Setup script for creating the data folder structure.

Usage:
    python scripts/setup_data_folder.py
"""

from pathlib import Path

# Hard-coded data root
DATA_ROOT = Path("data")


def setup_data_folder():
    """Create the data/AdVitam folder."""

    advitam_path = DATA_ROOT / "AdVitam"

    if advitam_path.exists():
        print("Already exists")
        return

    advitam_path.mkdir(parents=True, exist_ok=True)
    print(f"Created folder structure under {advitam_path}")


if __name__ == "__main__":
    setup_data_folder()
