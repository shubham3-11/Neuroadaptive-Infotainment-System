#!/usr/bin/env python3
"""
Download script for the AdVitam Exp4 dataset.

Usage:
    python scripts/download_advitam.py [--force]
"""

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

# --- Hard-coded dataset directories ---------------------------------------
DATA_ROOT = Path("data")

ADVITAM_EXP4_ROOT = DATA_ROOT / "AdVitam" / "Exp4"


def download_advitam_exp4(force: bool = False):
    """Download and setup the AdVitam Exp4 dataset."""

    advitam_root = DATA_ROOT / "AdVitam"
    exp4_path = ADVITAM_EXP4_ROOT
    zip_path = advitam_root / "Exp4.zip"

    # Check if data/AdVitam exists
    if not advitam_root.exists():
        print(f"Error: {advitam_root} not found. Run setup_data_folder.py first.")
        sys.exit(1)

    # Check if dataset already exists (look for some files in Exp4)
    if exp4_path.exists() and not force:
        if any(exp4_path.rglob("*")):  # If there are any files
            print("Already exists")
            return

    # Download with progress bar
    url = "https://zenodo.org/records/7319612/files/Exp4.zip?download=1"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownloading: {progress:.1f}%", end="", flush=True)
                    else:
                        print(
                            f"\rDownloading: {downloaded // 1024 // 1024} MB",
                            end="",
                            flush=True,
                        )

        print()  # New line after progress
        print("Download complete")

    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)

    # Extract
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(advitam_root)
        zip_path.unlink()  # Remove zip file
        print("Extraction complete")
    except Exception as e:
        print(f"Error extracting: {e}")
        sys.exit(1)

    print("Done")


def main():
    parser = argparse.ArgumentParser(description="Download AdVitam Exp4 dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    try:
        download_advitam_exp4(force=args.force)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
