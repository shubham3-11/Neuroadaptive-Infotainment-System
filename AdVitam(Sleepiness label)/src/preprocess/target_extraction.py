#!/usr/bin/env python3
"""
Target Extraction for AdVitam Dataset - KSS Interpolation

This script reads the questionnaire data and generates interpolated KSS values
for each chunk based on the KSS_B_1, KSS_1, KSS_B_2, KSS_2 values.

Usage:
    python target_extraction.py
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import warnings
# --- Hard-coded dataset locations -----------------------------------------
ADVITAM_EXP4_QUESTIONNAIRE_FILE = Path("data/AdVitam/Preprocessed/Questionnaire/Exp4_Database.csv")

ADVITAM_EXP4_TARGET_DIR = Path("data/AdVitam/Preprocessed2/Target")

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdVitam KSS target extraction with flexible output root and chunk count"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(ADVITAM_EXP4_QUESTIONNAIRE_FILE),
        help="Path to questionnaire CSV (default: data/.../Exp4_Database.csv)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/AdVitam/Preprocessed2/Target",
        help="Root folder for saving targets. Final dir = output_root/n{chunks}",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=6,
        help="Number of interpolated chunks per scenario (default: 6)",
    )
    return parser.parse_args()


def load_questionnaire_data(file_path):
    """
    Load questionnaire data and extract KSS-related columns

    Parameters:
    -----------
    file_path : str
        Path to the questionnaire CSV file

    Returns:
    --------
    tuple
        (scenario1_df, scenario2_df) - DataFrames for each scenario
    """
    print(f"Loading questionnaire data from: {file_path}")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract relevant columns
    kss_columns = ["participant_code", "KSS_B_1", "KSS_1", "KSS_B_2", "KSS_2"]

    # Check if all required columns exist
    missing_columns = [col for col in kss_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        return None, None

    # Select only the required columns
    kss_df = df[kss_columns].copy()

    # Create separate datasets for each scenario
    # Scenario 1: KSS_B_1 and KSS_1
    scenario1_df = kss_df[["participant_code", "KSS_B_1", "KSS_1"]].copy()
    scenario1_df = scenario1_df.dropna(subset=["KSS_B_1", "KSS_1"])
    scenario1_df = scenario1_df.rename(
        columns={"KSS_B_1": "KSS_baseline", "KSS_1": "KSS_end"}
    )
    scenario1_df["scenario"] = 1

    # Scenario 2: KSS_B_2 and KSS_2
    scenario2_df = kss_df[["participant_code", "KSS_B_2", "KSS_2"]].copy()
    scenario2_df = scenario2_df.dropna(subset=["KSS_B_2", "KSS_2"])
    scenario2_df = scenario2_df.rename(
        columns={"KSS_B_2": "KSS_baseline", "KSS_2": "KSS_end"}
    )
    scenario2_df["scenario"] = 2

    print(f"Scenario 1: {len(scenario1_df)} participants with complete KSS data")
    print(f"Scenario 2: {len(scenario2_df)} participants with complete KSS data")

    return scenario1_df, scenario2_df


def interpolate_kss_for_chunks(kss_baseline, kss_end, n_chunks=6):
    """
    Interpolate KSS values for chunks between baseline and end measurements

    Parameters:
    -----------
    kss_baseline : float
        KSS baseline before the scenario
    kss_end : float
        KSS after the scenario
    n_chunks : int
        Number of chunks per scenario

    Returns:
    --------
    list
        List of interpolated KSS values for each chunk
    """
    # Linear interpolation between baseline and end
    kss_values = np.linspace(
        kss_baseline, kss_end, n_chunks + 1
    )  # +1 to include both endpoints
    chunk_values = kss_values[1:]  # Remove the baseline value, keep chunk values

    return chunk_values.tolist()


def generate_targets_for_scenario(scenario_df, n_chunks=6):
    """
    Generate interpolated KSS targets for a specific scenario

    Parameters:
    -----------
    scenario_df : pd.DataFrame
        DataFrame with KSS data for one scenario
    n_chunks : int
        Number of chunks per scenario

    Returns:
    --------
    dict
        Dictionary with participant codes as keys and interpolated KSS values
    """
    targets = {}

    for _, row in scenario_df.iterrows():
        participant_code = row["participant_code"]
        scenario = row["scenario"]

        # Extract KSS values
        kss_baseline = row["KSS_baseline"]
        kss_end = row["KSS_end"]

        # Interpolate KSS values for chunks
        chunk_values = interpolate_kss_for_chunks(kss_baseline, kss_end, n_chunks)

        targets[participant_code] = {
            "scenario": scenario,
            "baseline": kss_baseline,
            "end": kss_end,
            "KSS_0": kss_baseline,  # KSS value during baseline period
            "chunks": chunk_values,
        }

    return targets


def save_scenario_targets(scenario1_targets, scenario2_targets, output_dir):
    """
    Save interpolated KSS targets for both scenarios to a single JSON file

    Parameters:
    -----------
    scenario1_targets : dict
        Dictionary with interpolated KSS targets for scenario 1
    scenario2_targets : dict
        Dictionary with interpolated KSS targets for scenario 2
    output_dir : str
        Directory to save the files

    Returns:
    --------
    tuple
        (scenario1_count, scenario2_count) - Number of participants in each scenario
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Combine all targets into one dictionary
    all_targets = {}

    # Add scenario 1 targets
    for participant_code, target_data in scenario1_targets.items():
        all_targets[f"{participant_code}_scenario1"] = target_data

    # Add scenario 2 targets
    for participant_code, target_data in scenario2_targets.items():
        all_targets[f"{participant_code}_scenario2"] = target_data

    # Save all targets to a single JSON file
    all_targets_file = output_path / "all_kss_targets.json"
    with open(all_targets_file, "w") as f:
        json.dump(all_targets, f, indent=2)

    print(f"✓ Saved all targets to: {all_targets_file}")
    print(
        f"✓ Total entries: {len(all_targets)} ({len(scenario1_targets)} scenario 1 + {len(scenario2_targets)} scenario 2)"
    )

    return len(scenario1_targets), len(scenario2_targets)


def print_summary_statistics(scenario1_targets, scenario2_targets):
    """
    Print summary statistics of the interpolated KSS targets for both scenarios

    Parameters:
    -----------
    scenario1_targets : dict
        Dictionary with interpolated KSS targets for scenario 1
    scenario2_targets : dict
        Dictionary with interpolated KSS targets for scenario 2
    """
    print(f"\n{'='*60}")
    print("KSS TARGETS SUMMARY STATISTICS")
    print(f"{'='*60}")

    # Collect all KSS values for each scenario
    scenario1_values = []
    scenario1_chunks = []
    scenario2_values = []
    scenario2_chunks = []

    for participant_code, target_data in scenario1_targets.items():
        scenario1_values.extend([target_data["baseline"], target_data["end"]])
        scenario1_chunks.extend(target_data["chunks"])

    for participant_code, target_data in scenario2_targets.items():
        scenario2_values.extend([target_data["baseline"], target_data["end"]])
        scenario2_chunks.extend(target_data["chunks"])

    scenario1_values = np.array(scenario1_values)
    scenario1_chunks = np.array(scenario1_chunks)
    scenario2_values = np.array(scenario2_values)
    scenario2_chunks = np.array(scenario2_chunks)

    print(f"Scenario 1 participants: {len(scenario1_targets)}")
    print(f"Scenario 2 participants: {len(scenario2_targets)}")
    print(f"Total participants: {len(scenario1_targets) + len(scenario2_targets)}")

    print(f"\nScenario 1 Statistics:")
    print(f"  KSS range: {scenario1_values.min():.1f} - {scenario1_values.max():.1f}")
    print(f"  KSS mean: {scenario1_values.mean():.2f}")
    print(f"  KSS std: {scenario1_values.std():.2f}")
    print(f"  Chunk range: {scenario1_chunks.min():.1f} - {scenario1_chunks.max():.1f}")
    print(f"  Chunk mean: {scenario1_chunks.mean():.2f}")
    print(f"  Chunk std: {scenario1_chunks.std():.2f}")

    print(f"\nScenario 2 Statistics:")
    print(f"  KSS range: {scenario2_values.min():.1f} - {scenario2_values.max():.1f}")
    print(f"  KSS mean: {scenario2_values.mean():.2f}")
    print(f"  KSS std: {scenario2_values.std():.2f}")
    print(f"  Chunk range: {scenario2_chunks.min():.1f} - {scenario2_chunks.max():.1f}")
    print(f"  Chunk mean: {scenario2_chunks.mean():.2f}")
    print(f"  Chunk std: {scenario2_chunks.std():.2f}")

    # Show sample interpolations
    print(f"\nSample Interpolations (first 2 participants per scenario):")

    print(f"\n  Scenario 1:")
    for i, (participant_code, target_data) in enumerate(
        list(scenario1_targets.items())[:2]
    ):
        print(
            f"    {participant_code}: KSS_0={target_data['KSS_0']:.1f}, {target_data['baseline']:.1f} → {target_data['chunks']} → {target_data['end']:.1f}"
        )

    print(f"\n  Scenario 2:")
    for i, (participant_code, target_data) in enumerate(
        list(scenario2_targets.items())[:2]
    ):
        print(
            f"    {participant_code}: KSS_0={target_data['KSS_0']:.1f}, {target_data['baseline']:.1f} → {target_data['chunks']} → {target_data['end']:.1f}"
        )


def main():
    """Main function to extract and interpolate KSS targets for both scenarios"""

    args = parse_args()

    questionnaire_file = args.input_file
    output_dir = str(Path(args.output_root) / f"n{args.n_chunks}")
    n_chunks = args.n_chunks

    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-targets_n{n_chunks}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
    )

    class _LoggerWriter:
        def __init__(self, lvl):
            self.lvl = lvl
            self.logger = logging.getLogger("stdout")
        def write(self, msg):
            msg = msg.strip()
            if msg:
                self.logger.log(self.lvl, msg)
        def flush(self):
            pass

    import sys
    sys.stdout = _LoggerWriter(logging.INFO)

    print("=== AdVitam KSS Target Extraction and Interpolation ===")
    print(f"Questionnaire file: {questionnaire_file}")
    print(f"Output directory: {output_dir}")
    print(f"Chunks per scenario: {n_chunks}")
    print()

    # Load questionnaire data
    scenario1_df, scenario2_df = load_questionnaire_data(questionnaire_file)

    if scenario1_df is None or scenario2_df is None:
        print("✗ Failed to load questionnaire data")
        return

    # Progress display
    print("\nGenerating interpolated KSS targets...")
    for _ in tqdm(range(2), desc="Scenarios", leave=False):
        pass  # just a visual placeholder; actual work already done

    # Generate interpolated targets for scenario 1
    scenario1_targets = generate_targets_for_scenario(scenario1_df, n_chunks)

    # Generate interpolated targets for scenario 2
    scenario2_targets = generate_targets_for_scenario(scenario2_df, n_chunks)

    if not scenario1_targets and not scenario2_targets:
        print("✗ No targets generated")
        return

    # Save targets
    print("\nSaving targets...")
    scenario1_count, scenario2_count = save_scenario_targets(
        scenario1_targets, scenario2_targets, output_dir
    )

    # Print summary statistics
    print_summary_statistics(scenario1_targets, scenario2_targets)

    print(f"\n{'='*60}")
    print("TARGET EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Generated KSS targets for {scenario1_count} scenario 1 participants")
    print(f"✓ Generated KSS targets for {scenario2_count} scenario 2 participants")
    print(f"✓ Interpolated {n_chunks} chunks per scenario")
    print(f"✓ Saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - All targets saved to a single JSON file")
    print(f"  - Includes KSS_0 (baseline KSS values) for each participant")
    print(f"\n=== Ready for LSTM/CNN+RNN Models ===")


if __name__ == "__main__":
    main()
