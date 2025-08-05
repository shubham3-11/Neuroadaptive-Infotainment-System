#!/usr/bin/env python3
"""
Feature Extraction for AdVitam Dataset - LSTM/CNN+RNN Models

This script extracts features from raw BioPac .acq files and creates matrices
suitable for LSTM and CNN+RNN models with windowed features.

Usage:
    python feature_extraction.py
"""

import bioread
import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path
import warnings
import json
import math
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Global helper for redirecting print -> logging
# ---------------------------------------------------------------------------


class _LoggerWriter:
    """File-like wrapper to funnel prints into logging."""

    def __init__(self, lvl):
        self.lvl = lvl
        self.logger = logging.getLogger("stdout")

    def write(self, msg):
        msg = msg.strip()
        if msg:
            self.logger.log(self.lvl, msg)

    def flush(self):
        pass


# --- Hard-coded dataset locations -----------------------------------------
# Raw BioPac recordings directory (Exp4)
ADVITAM_EXP4_RAW_BIOPAC_DIR = Path("data/AdVitam/Raw/Physio/BioPac")

# Pre-extracted windowed features directory
ADVITAM_EXP4_FEATURE_DIR = Path("data/AdVitam/Preprocessed2/Feature")

warnings.filterwarnings("ignore")


def read_bioac_file(file_path):
    """Read BioPac .acq file and extract signals, event markers, and sampling rate"""
    print(f"Reading: {file_path}")

    try:
        data = bioread.read_file(file_path)
        signals = {}

        for channel in data.channels:
            name = channel.name
            if "EDA" in name:
                signals["eda"] = channel.data
            elif "ECG" in name:
                signals["ecg"] = channel.data
            elif "Respiration" in name or "RSP" in name:
                signals["respiration"] = channel.data

        print(f"Found signals: {list(signals.keys())}")

        # Extract event markers
        event_markers = []
        if hasattr(data, "event_markers") and data.event_markers:
            for marker in data.event_markers:
                # Calculate time in seconds from sample index
                time_in_seconds = marker.sample_index / data.samples_per_second
                event_markers.append(
                    {
                        "name": marker.text,
                        "time": time_in_seconds,
                        "sample_index": marker.sample_index,
                    }
                )
            print(f"Found {len(event_markers)} event markers")
            for marker in event_markers:
                print(
                    f"  - {marker['name']}: {marker['time']:.1f}s ({marker['sample_index']})"
                )

        # Get sampling rate from file
        sampling_rate = data.samples_per_second
        print(f"Detected sampling rate: {sampling_rate} Hz")

        return signals, event_markers, sampling_rate

    except Exception as e:
        print(f"Error reading file: {e}")
        return {}, [], None


def extract_features(signals, sampling_rate=1000):
    """Extract features from physiological signals"""
    features = {}

    # EDA features
    if "eda" in signals:
        try:
            # Validate EDA signal before processing
            eda_signal = signals["eda"]
            # EDA needs at least 1 second of data for meaningful processing
            min_eda_samples = sampling_rate * 1  # 1 second minimum
            if len(eda_signal) < min_eda_samples:
                print(
                    f"EDA signal too short ({len(eda_signal)} samples < {min_eda_samples}), skipping"
                )
                features["EDA_filtered_mean"] = np.nan
                features["EDA_filtered_std"] = np.nan
                features["EDA_filtered_min"] = np.nan
                features["EDA_filtered_max"] = np.nan
                features["EDA_tonic_mean"] = np.nan
                features["EDA_tonic_std"] = np.nan
                features["EDA_tonic_min"] = np.nan
                features["EDA_tonic_max"] = np.nan
                features["SCR_Peaks_freq"] = np.nan
                features["SCR_Peaks_N"] = np.nan
                features["SCR_Peaks_Amplitude_Mean"] = np.nan
            else:
                eda_processed, _ = nk.eda_process(
                    eda_signal, sampling_rate=sampling_rate
                )

                if "EDA_Filtered" in eda_processed.columns:
                    filtered = eda_processed["EDA_Filtered"].dropna()
                    features["EDA_filtered_mean"] = (
                        filtered.mean() if len(filtered) > 0 else np.nan
                    )
                    features["EDA_filtered_std"] = (
                        filtered.std() if len(filtered) > 0 else np.nan
                    )
                    features["EDA_filtered_min"] = (
                        filtered.min() if len(filtered) > 0 else np.nan
                    )
                    features["EDA_filtered_max"] = (
                        filtered.max() if len(filtered) > 0 else np.nan
                    )

                if "EDA_Tonic" in eda_processed.columns:
                    tonic = eda_processed["EDA_Tonic"].dropna()
                    features["EDA_tonic_mean"] = (
                        tonic.mean() if len(tonic) > 0 else np.nan
                    )
                    features["EDA_tonic_std"] = (
                        tonic.std() if len(tonic) > 0 else np.nan
                    )
                    features["EDA_tonic_min"] = (
                        tonic.min() if len(tonic) > 0 else np.nan
                    )
                    features["EDA_tonic_max"] = (
                        tonic.max() if len(tonic) > 0 else np.nan
                    )

                if "SCR_Peaks" in eda_processed.columns:
                    peaks = eda_processed["SCR_Peaks"].dropna()
                    features["SCR_Peaks_freq"] = (
                        len(peaks) / (len(eda_signal) / sampling_rate / 60)
                        if len(peaks) > 0
                        else 0
                    )
                    features["SCR_Peaks_N"] = len(peaks)
                    features["SCR_Peaks_Amplitude_Mean"] = (
                        peaks.mean() if len(peaks) > 0 else 0
                    )

        except Exception as e:
            print(f"EDA processing error: {e}")
            # Set default values for EDA features
            features["EDA_filtered_mean"] = np.nan
            features["EDA_filtered_std"] = np.nan
            features["EDA_filtered_min"] = np.nan
            features["EDA_filtered_max"] = np.nan
            features["EDA_tonic_mean"] = np.nan
            features["EDA_tonic_std"] = np.nan
            features["EDA_tonic_min"] = np.nan
            features["EDA_tonic_max"] = np.nan
            features["SCR_Peaks_freq"] = np.nan
            features["SCR_Peaks_N"] = np.nan
            features["SCR_Peaks_Amplitude_Mean"] = np.nan

    # ECG features
    if "ecg" in signals:
        try:
            # Validate ECG signal before processing
            ecg_signal = signals["ecg"]
            # ECG needs at least 2 seconds for reliable R-peak detection and HRV
            min_ecg_samples = sampling_rate * 2  # 2 seconds minimum
            if len(ecg_signal) < min_ecg_samples:
                print(
                    f"ECG signal too short ({len(ecg_signal)} samples < {min_ecg_samples}), skipping"
                )
                features["ECG_Rate_Mean"] = np.nan
                # Add default values for all HRV features
                hrv_features = [
                    "HRV_MeanNN",
                    "HRV_SDNN",
                    "HRV_SDANN1",
                    "HRV_SDNNI1",
                    "HRV_RMSSD",
                    "HRV_SDSD",
                    "HRV_CVNN",
                    "HRV_CVSD",
                    "HRV_MedianNN",
                    "HRV_MadNN",
                    "HRV_MCVNN",
                    "HRV_IQRNN",
                    "HRV_pNN50",
                    "HRV_pNN20",
                    "HRV_HTI",
                    "HRV_TINN",
                    "HRV_LF",
                    "HRV_HF",
                    "HRV_VHF",
                    "HRV_LFHF",
                    "HRV_LFn",
                    "HRV_HFn",
                    "HRV_LnHF",
                    "HRV_SD1",
                    "HRV_SD2",
                    "HRV_SD1SD2",
                    "HRV_S",
                    "HRV_CSI",
                    "HRV_CVI",
                    "HRV_CSI_Modified",
                    "HRV_PIP",
                    "HRV_IALS",
                    "HRV_PSS",
                    "HRV_PAS",
                    "HRV_GI",
                    "HRV_SI",
                    "HRV_AI",
                    "HRV_PI",
                    "HRV_C1d",
                    "HRV_C1a",
                    "HRV_SD1d",
                    "HRV_SD1a",
                    "HRV_C2d",
                    "HRV_C2a",
                    "HRV_SD2d",
                    "HRV_SD2a",
                    "HRV_Cd",
                    "HRV_Ca",
                    "HRV_SDNNd",
                    "HRV_SDNNa",
                ]
                for feature in hrv_features:
                    features[feature] = np.nan
            else:
                ecg_processed, _ = nk.ecg_process(
                    ecg_signal, sampling_rate=sampling_rate
                )

                if "ECG_Rate" in ecg_processed.columns:
                    rate = ecg_processed["ECG_Rate"].dropna()
                    features["ECG_Rate_Mean"] = rate.mean() if len(rate) > 0 else np.nan

                # HRV features
                hrv_time = nk.hrv_time(ecg_processed, sampling_rate=sampling_rate)
                hrv_freq = nk.hrv_frequency(ecg_processed, sampling_rate=sampling_rate)

                for col in hrv_time.columns:
                    if col != "HRV_MeanNN":
                        features[col] = (
                            hrv_time[col].iloc[0] if len(hrv_time) > 0 else np.nan
                        )

                for col in hrv_freq.columns:
                    features[col] = (
                        hrv_freq[col].iloc[0] if len(hrv_freq) > 0 else np.nan
                    )

        except Exception as e:
            print(f"ECG processing error: {e}")
            # Set default values for ECG features
            features["ECG_Rate_Mean"] = np.nan
            # Add default values for all HRV features
            hrv_features = [
                "HRV_MeanNN",
                "HRV_SDNN",
                "HRV_SDANN1",
                "HRV_SDNNI1",
                "HRV_RMSSD",
                "HRV_SDSD",
                "HRV_CVNN",
                "HRV_CVSD",
                "HRV_MedianNN",
                "HRV_MadNN",
                "HRV_MCVNN",
                "HRV_IQRNN",
                "HRV_pNN50",
                "HRV_pNN20",
                "HRV_HTI",
                "HRV_TINN",
                "HRV_LF",
                "HRV_HF",
                "HRV_VHF",
                "HRV_LFHF",
                "HRV_LFn",
                "HRV_HFn",
                "HRV_LnHF",
                "HRV_SD1",
                "HRV_SD2",
                "HRV_SD1SD2",
                "HRV_S",
                "HRV_CSI",
                "HRV_CVI",
                "HRV_CSI_Modified",
                "HRV_PIP",
                "HRV_IALS",
                "HRV_PSS",
                "HRV_PAS",
                "HRV_GI",
                "HRV_SI",
                "HRV_AI",
                "HRV_PI",
                "HRV_C1d",
                "HRV_C1a",
                "HRV_SD1d",
                "HRV_SD1a",
                "HRV_C2d",
                "HRV_C2a",
                "HRV_SD2d",
                "HRV_SD2a",
                "HRV_Cd",
                "HRV_Ca",
                "HRV_SDNNd",
                "HRV_SDNNa",
            ]
            for feature in hrv_features:
                features[feature] = np.nan

    # Respiration features
    if "respiration" in signals:
        try:
            # Validate respiration signal before processing
            resp_signal = signals["respiration"]
            # Respiration needs at least 3 seconds for one complete breathing cycle
            min_resp_samples = sampling_rate * 3  # 3 seconds minimum
            if len(resp_signal) < min_resp_samples:
                print(
                    f"Respiration signal too short ({len(resp_signal)} samples < {min_resp_samples}), skipping"
                )
                features["RSP_Rate_Mean"] = np.nan
                features["RSP_Amplitude_Mean"] = np.nan
            else:
                resp_processed, _ = nk.rsp_process(
                    resp_signal, sampling_rate=sampling_rate
                )

                if "RSP_Rate" in resp_processed.columns:
                    rate = resp_processed["RSP_Rate"].dropna()
                    features["RSP_Rate_Mean"] = rate.mean() if len(rate) > 0 else np.nan

                if "RSP_Amplitude" in resp_processed.columns:
                    amplitude = resp_processed["RSP_Amplitude"].dropna()
                    features["RSP_Amplitude_Mean"] = (
                        amplitude.mean() if len(amplitude) > 0 else np.nan
                    )

        except Exception as e:
            print(f"Respiration processing error: {e}")
            # Set default values for respiration features
            features["RSP_Rate_Mean"] = np.nan
            features["RSP_Amplitude_Mean"] = np.nan

    return features


def extract_baseline_period_from_markers(event_markers, sampling_rate=1000):
    """
    Extract baseline period from event markers

    Parameters:
    -----------
    event_markers : list
        List of event marker dictionaries
    sampling_rate : int
        Sampling rate in Hz

    Returns:
    --------
    dict or None
        Baseline period dictionary with start/end times and sample indices, or None if not found
    """
    # Look for baseline start/end markers
    baseline_starts = [
        m
        for m in event_markers
        if "baseline" in m["name"].lower() and "start" in m["name"].lower()
    ]
    baseline_ends = [
        m
        for m in event_markers
        if "baseline" in m["name"].lower() and "end" in m["name"].lower()
    ]

    # Sort by time
    baseline_starts.sort(key=lambda x: x["time"])
    baseline_ends.sort(key=lambda x: x["time"])

    print(
        f"\nFound {len(baseline_starts)} baseline starts and {len(baseline_ends)} baseline ends"
    )

    # Match start and end markers
    for i, start_marker in enumerate(baseline_starts):
        if i < len(baseline_ends):
            end_marker = baseline_ends[i]

            # Ensure end comes after start
            if end_marker["time"] > start_marker["time"]:
                baseline_period = {
                    "name": f"Baseline_{i+1}",
                    "start_time": start_marker["time"],
                    "end_time": end_marker["time"],
                    "start_sample": start_marker["sample_index"],
                    "end_sample": end_marker["sample_index"],
                    "duration_seconds": end_marker["time"] - start_marker["time"],
                    "duration_minutes": (end_marker["time"] - start_marker["time"])
                    / 60,
                }
                print(
                    f"  ✓ Baseline {i+1}: {start_marker['time']:.1f}s - {end_marker['time']:.1f}s "
                    f"({baseline_period['duration_minutes']:.1f} minutes)"
                )
                return baseline_period
            else:
                print(f"  ✗ Baseline {i+1}: Invalid timing (end before start)")

    print("  ✗ No valid baseline period found")
    return None


def extract_baseline_features(signals, baseline_period, sampling_rate=1000):
    """
    Extract baseline features from the baseline period

    Parameters:
    -----------
    signals : dict
        Dictionary of physiological signals
    baseline_period : dict
        Baseline period information with start/end sample indices
    sampling_rate : int
        Sampling rate in Hz

    Returns:
    --------
    dict
        Dictionary of baseline features
    """
    if baseline_period is None:
        print("No baseline period provided, skipping baseline feature extraction")
        return {}

    start_sample = baseline_period["start_sample"]
    end_sample = baseline_period["end_sample"]

    print(f"\nExtracting baseline features from samples {start_sample} to {end_sample}")
    print(f"Baseline duration: {baseline_period['duration_minutes']:.1f} minutes")

    # Extract baseline signals
    baseline_signals = {}
    for signal_name, signal_data in signals.items():
        baseline_signals[signal_name] = signal_data[start_sample:end_sample]

    # Extract features from baseline period
    baseline_features = extract_features(baseline_signals, sampling_rate)

    # Add baseline period information to feature names
    baseline_features_with_prefix = {}
    for feature_name, feature_value in baseline_features.items():
        baseline_features_with_prefix[f"baseline_{feature_name}"] = feature_value

    # Add baseline period metadata
    baseline_features_with_prefix["baseline_duration_seconds"] = baseline_period[
        "duration_seconds"
    ]
    baseline_features_with_prefix["baseline_duration_minutes"] = baseline_period[
        "duration_minutes"
    ]
    baseline_features_with_prefix["baseline_start_time"] = baseline_period["start_time"]
    baseline_features_with_prefix["baseline_end_time"] = baseline_period["end_time"]

    print(f"✓ Extracted {len(baseline_features_with_prefix)} baseline features")

    return baseline_features_with_prefix


def extract_block_periods_from_markers(event_markers, sampling_rate=1000):
    """
    Extract block periods from event markers

    Parameters:
    -----------
    event_markers : list
        List of event marker dictionaries
    sampling_rate : int
        Sampling rate in Hz

    Returns:
    --------
    list
        List of block period dictionaries
    """
    block_periods = []

    # Look for block start/end markers
    block_starts = [
        m
        for m in event_markers
        if "block" in m["name"].lower() and "start" in m["name"].lower()
    ]
    block_ends = [
        m
        for m in event_markers
        if "block" in m["name"].lower() and "end" in m["name"].lower()
    ]

    # Sort by time
    block_starts.sort(key=lambda x: x["time"])
    block_ends.sort(key=lambda x: x["time"])

    print(f"\nFound {len(block_starts)} block starts and {len(block_ends)} block ends")

    # Match start and end markers
    for i, start_marker in enumerate(block_starts):
        if i < len(block_ends):
            end_marker = block_ends[i]

            # Ensure end comes after start
            if end_marker["time"] > start_marker["time"]:
                block_periods.append(
                    {
                        "name": f"Block_{i+1}",
                        "start_time": start_marker["time"],
                        "end_time": end_marker["time"],
                        "start_sample": start_marker["sample_index"],
                        "end_sample": end_marker["sample_index"],
                    }
                )
                print(
                    f"  ✓ Block {i+1}: {start_marker['time']:.1f}s - {end_marker['time']:.1f}s"
                )
            else:
                print(f"  ✗ Block {i+1}: Invalid timing (end before start)")

    return block_periods


def create_windowed_feature_matrix(
    file_path,
    window_duration=30,
    slide=10,
    chunk_duration=300,
    n_chunks=5,
    include_baseline=True,
):
    """
    Create feature matrix with exactly n_chunks of fixed duration per block, each chunk containing
    multiple windows based on window_duration and slide parameters

    Parameters:
    -----------
    file_path : str
        Path to BioPac .acq file
    window_duration : int
        Duration of each window in seconds
    slide : int
        Slide/step size between windows in seconds (overlap = window_duration - slide)
    chunk_duration : int
        Duration of each chunk in seconds (default: 300 seconds = 5 minutes)
    n_chunks : int
        Number of chunks per block (default: 5)
    include_baseline : bool
        Whether to include baseline features (default: True)

    Returns:
    --------
    tuple : (feature_matrices, feature_names, block_info, sampling_rate, baseline_features)
        - feature_matrices: list of numpy arrays, each with shape (n_chunks, n_windows_per_chunk, n_features) per block
        - feature_names: list of feature names
        - block_info: list of block information with chunk and window details
        - sampling_rate: detected sampling rate from file
        - baseline_features: dictionary of baseline features (if include_baseline=True)
    """
    # Read signals, event markers, and sampling rate
    signals, event_markers, sampling_rate = read_bioac_file(file_path)
    if not signals or sampling_rate is None:
        return [], [], [], None, {}

    # Extract baseline period and features if requested
    baseline_features = {}
    if include_baseline:
        baseline_period = extract_baseline_period_from_markers(
            event_markers, sampling_rate
        )
        if baseline_period:
            baseline_features = extract_baseline_features(
                signals, baseline_period, sampling_rate
            )
        else:
            print(
                "[WARNING] No baseline period found, continuing without baseline features"
            )

    # Extract block periods from event markers
    block_periods = extract_block_periods_from_markers(event_markers, sampling_rate)

    if not block_periods:
        print(
            "[ERROR] No valid block periods found in event markers. Skipping preprocessing for this file."
        )
        return [], [], [], sampling_rate, baseline_features

    # Get minimum signal length
    min_length = min(len(signal) for signal in signals.values())

    # Calculate window parameters in samples
    window_samples = int(window_duration * sampling_rate)
    slide_samples = int(slide * sampling_rate)
    overlap_samples = window_samples - slide_samples

    # Calculate chunk parameters in samples
    chunk_samples = int(chunk_duration * sampling_rate)

    print(f"Number of chunks per block: {n_chunks}")
    print(f"Chunk duration: {chunk_duration}s ({chunk_samples} samples)")
    print(f"Window duration: {window_duration}s ({window_samples} samples)")
    print(f"Slide: {slide}s ({slide_samples} samples)")
    print(f"Overlap: {overlap_samples} samples")

    # Create blocks with n_chunks of fixed duration, each chunk with multiple windows
    feature_matrices = []
    block_info = []
    feature_names = None

    for block_period in block_periods:
        start_idx = block_period["start_sample"]
        end_idx = block_period["end_sample"]
        block_name = block_period.get("name", "UnknownBlock")

        print(f"\nProcessing {block_name}...")

        # Ensure we don't exceed signal length
        if end_idx > min_length:
            print(
                f"[SKIP] {block_name}: end index {end_idx} exceeds signal length {min_length}. Skipping block."
            )
            continue
        if start_idx >= end_idx:
            print(
                f"[SKIP] {block_name}: start index {start_idx} >= end index {end_idx}. Skipping block."
            )
            continue

        # Calculate total block duration and check if it can accommodate n_chunks
        block_duration = end_idx - start_idx
        total_chunk_samples = n_chunks * chunk_samples

        # Calculate how many complete chunks we can fit
        complete_chunks = min(n_chunks, block_duration // chunk_samples)

        if complete_chunks == 0:
            print(
                f"[SKIP] {block_name}: Block too short ({block_duration} samples) for even 1 chunk of {chunk_samples} samples. Skipping block."
            )
            continue

        # Check if chunks are long enough for at least one window
        if chunk_samples < window_samples:
            print(
                f"[SKIP] {block_name}: Chunk size ({chunk_samples} samples) too small for window size ({window_samples} samples). Skipping block."
            )
            continue

        print(f"  Block duration: {block_duration/sampling_rate/60:.1f} minutes")
        print(
            f"  Can fit {complete_chunks} complete chunks out of {n_chunks} requested"
        )
        if complete_chunks < n_chunks:
            print(f"  Will create only {complete_chunks} chunks (no padding)")

        # Create only the chunks that fit within this block
        block_chunks = []
        chunk_info = []

        for chunk_idx in range(complete_chunks):
            # Create a real chunk
            chunk_start = start_idx + chunk_idx * chunk_samples
            chunk_end = chunk_start + chunk_samples

            # Create windows within this chunk
            chunk_windows = []
            window_info = []
            window_start = chunk_start

            while window_start + window_samples <= chunk_end:
                window_end = window_start + window_samples

                # Extract window for each signal
                window_signals = {}
                for signal_name, signal_data in signals.items():
                    window_signals[signal_name] = signal_data[window_start:window_end]

                # Extract features for this window
                window_features = extract_features(window_signals, sampling_rate)

                if feature_names is None:
                    feature_names = list(window_features.keys())

                chunk_windows.append(list(window_features.values()))

                # Store window information
                window_info.append(
                    {
                        "window_start_sample": window_start,
                        "window_end_sample": window_end,
                        "window_start_time": window_start / sampling_rate,
                        "window_end_time": window_end / sampling_rate,
                        "window_duration": window_duration,
                        "slide": slide,
                    }
                )

                # Move to next window
                window_start += slide_samples

            if chunk_windows:  # Only add if we have at least one window
                # Convert chunk_windows to numpy array
                chunk_matrix = np.array(chunk_windows)
                block_chunks.append(chunk_matrix)

                chunk_info.append(
                    {
                        "chunk_idx": chunk_idx,
                        "chunk_start_sample": chunk_start,
                        "chunk_end_sample": chunk_end,
                        "chunk_start_time": chunk_start / sampling_rate,
                        "chunk_end_time": chunk_end / sampling_rate,
                        "chunk_duration": chunk_duration,
                        "n_windows": len(chunk_windows),
                        "windows": window_info,
                    }
                )
            else:
                print(
                    f"[SKIP] Chunk {chunk_idx+1}: No valid windows created. Skipping chunk."
                )

        if len(block_chunks) > 0:  # Only add if we have at least one chunk
            # Stack chunks to create block matrix: (n_actual_chunks, n_windows_per_chunk, n_features)
            # Note: chunks may have different numbers of windows, so we'll handle this carefully
            max_windows = max(chunk.shape[0] for chunk in block_chunks)
            n_features = len(feature_names) if feature_names else 0

            # Create block matrix with actual number of chunks
            actual_chunks = len(block_chunks)
            block_matrix = np.full((actual_chunks, max_windows, n_features), np.nan)
            for chunk_idx, chunk_matrix in enumerate(block_chunks):
                n_windows = chunk_matrix.shape[0]
                block_matrix[chunk_idx, :n_windows, :] = chunk_matrix

            feature_matrices.append(block_matrix)

            block_info.append(
                {
                    "name": block_name,
                    "start_time": block_period["start_time"],
                    "end_time": block_period["end_time"],
                    "start_sample": start_idx,
                    "end_sample": end_idx,
                    "n_chunks": actual_chunks,
                    "requested_chunks": n_chunks,
                    "chunk_duration": chunk_duration,
                    "window_duration": window_duration,
                    "slide": slide,
                    "chunks": chunk_info,
                }
            )

            # Count total windows
            total_windows_in_block = sum(chunk["n_windows"] for chunk in chunk_info)
            print(
                f"✓ Created {actual_chunks} chunks for {block_name} (out of {n_chunks} requested)"
            )
            print(
                f"  Total windows: {total_windows_in_block}, Max windows per chunk: {max_windows}"
            )

            # Print window count per chunk
            for chunk_idx, chunk in enumerate(chunk_info):
                print(f"  Chunk {chunk_idx+1}: {chunk['n_windows']} windows")
        else:
            print(f"[SKIP] {block_name}: No valid chunks created. Skipping block.")

    if not feature_matrices:
        print(
            "[ERROR] No valid blocks found after chunking and windowing. No features extracted."
        )
        return [], [], [], sampling_rate, baseline_features

    return (
        feature_matrices,
        feature_names,
        block_info,
        sampling_rate,
        baseline_features,
    )


def process_single_file(
    file_path, output_dir, window_duration=30, slide=10, n_chunks=6
):
    """
    Process a single BioPac file and save results to output directory

    Parameters:
    -----------
    file_path : str
        Path to the .acq file
    output_dir : str
        Directory to save output files
    window_duration : int
        Window duration in seconds
    slide : int
        Slide duration in seconds
    n_chunks : int
        Number of chunks per block

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Extract participant ID from filename
        participant_id = Path(file_path).stem  # e.g., "01_AC16" from "01_AC16.acq"

        print(f"\n{'='*60}")
        print(f"Processing: {participant_id}")
        print(f"{'='*60}")

        expected_duration = 30 * 60  # 30 minutes
        chunk_duration = expected_duration / n_chunks  # 5 minutes

        # Create windowed feature matrix for the driving blocks
        print("1. Creating windowed feature matrix...")
        (
            feature_matrices,
            feature_names,
            block_info,
            sampling_rate,
            baseline_features,
        ) = create_windowed_feature_matrix(
            file_path,
            window_duration=window_duration,
            slide=slide,
            chunk_duration=chunk_duration,
            n_chunks=n_chunks,
            include_baseline=True,
        )

        if len(feature_matrices) > 0:
            print(f"✓ Successfully extracted features for {participant_id}!")
            print(f"  - Number of blocks: {len(feature_matrices)}")
            print(f"  - Number of features: {len(feature_names)}")
            print(f"  - Detected sampling rate: {sampling_rate} Hz")
            if baseline_features:
                print(f"  - Baseline features: {len(baseline_features)} features")
            else:
                print(f"  - Baseline features: None available")

            # Save results to output directory
            for i, feature_matrix in enumerate(feature_matrices):
                output_file = (
                    Path(output_dir) / f"{participant_id}_windowed_features_{i+1}.npy"
                )
                np.save(output_file, feature_matrix)
                print(f"  ✓ Saved: {output_file}")

            # Save feature names
            feature_names_file = (
                Path(output_dir) / f"{participant_id}_feature_names.txt"
            )
            with open(feature_names_file, "w") as f:
                for name in feature_names:
                    f.write(f"{name}\n")
            print(f"  ✓ Saved: {feature_names_file}")

            # Save block information
            block_info_file = Path(output_dir) / f"{participant_id}_block_info.json"
            with open(block_info_file, "w") as f:
                json.dump(block_info, f, indent=2)
            print(f"  ✓ Saved: {block_info_file}")

            # Save baseline features if available
            if baseline_features:
                baseline_file = (
                    Path(output_dir) / f"{participant_id}_baseline_features.json"
                )
                with open(baseline_file, "w") as f:
                    json.dump(clean_nans(baseline_features), f, indent=2)
                print(f"  ✓ Saved: {baseline_file}")
                print(
                    f"  Baseline features: {len(baseline_features)} features extracted"
                )
            else:
                print(f"  ⚠️  No baseline features available for {participant_id}")

            # Print summary statistics
            total_windows = 0
            for i, (feature_matrix, block) in enumerate(
                zip(feature_matrices, block_info)
            ):
                total_windows += sum(chunk["n_windows"] for chunk in block["chunks"])
                print(f"  Block {i+1}: {block['name']} - {block['n_chunks']} chunks")

            print(f"  Total windows: {total_windows}")
            return True

        else:
            print(f"✗ Failed to extract features for {participant_id}")
            return False

    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False


def create_baseline_summary(output_dir):
    """
    Create a summary of all baseline features across participants

    Parameters:
    -----------
    output_dir : str
        Directory containing baseline feature files

    Returns:
    --------
    dict
        Summary statistics of baseline features across all participants
    """
    output_path = Path(output_dir)
    baseline_files = list(output_path.glob("*_baseline_features.json"))

    if not baseline_files:
        print("No baseline feature files found")
        return {}

    print(f"\nCreating baseline summary from {len(baseline_files)} files...")

    all_baseline_features = []
    participant_ids = []

    for baseline_file in baseline_files:
        participant_id = baseline_file.stem.replace("_baseline_features", "")
        participant_ids.append(participant_id)

        try:
            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)
            all_baseline_features.append(baseline_data)
            print(f"  ✓ Loaded baseline features for {participant_id}")
        except Exception as e:
            print(f"  ✗ Error loading {baseline_file}: {e}")

    if not all_baseline_features:
        print("No valid baseline features found")
        return {}

    # Create summary statistics
    feature_names = list(all_baseline_features[0].keys())
    summary = {
        "n_participants": len(participant_ids),
        "participant_ids": participant_ids,
        "n_baseline_features": len(feature_names),
        "baseline_feature_names": feature_names,
        "summary_statistics": {},
    }

    # Calculate statistics for each feature
    for feature_name in feature_names:
        if feature_name.startswith("baseline_"):
            values = []
            for baseline_data in all_baseline_features:
                if feature_name in baseline_data:
                    value = baseline_data[feature_name]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        values.append(value)

            if values:
                summary["summary_statistics"][feature_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_valid": int(len(values)),
                }

    # Save summary
    summary_file = output_path / "baseline_features_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Baseline summary saved to: {summary_file}")
    print(f"  - {len(participant_ids)} participants")
    print(f"  - {len(feature_names)} baseline features")
    print(f"  - {len(summary['summary_statistics'])} features with valid statistics")

    return summary


def clean_nans(obj):
    """Recursively replace NaN/inf/-inf with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    else:
        return obj


def parse_args():
    """Parse command-line arguments for dynamic preprocessing."""
    parser = argparse.ArgumentParser(description="AdVitam feature extraction with flexible window/slide parameters")
    parser.add_argument("--input-dir", type=str, default=str(ADVITAM_EXP4_RAW_BIOPAC_DIR), help="Directory containing .acq files (default: data/AdVitam/Raw/Physio/BioPac)")
    parser.add_argument("--output-root", type=str, default="data/AdVitam/Preprocessed2", help="Root folder where preprocessed features will be stored. Final directory is output_root/w{window}_s{slide}_n{chunks}")
    parser.add_argument("--window", type=int, default=30, help="Window duration in seconds (default: 30)")
    parser.add_argument("--slide", type=int, default=10, help="Slide/step between windows in seconds (default: 10)")
    parser.add_argument("--n-chunks", type=int, default=6, help="Number of 5-minute chunks per driving block (default: 6)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--failed-filename", type=str, default=None, help="Optional path to a text file with failed filenames to reprocess")
    return parser.parse_args()


def main(failed_filename=None):
    # Parse CLI args (overrides defaults)
    args = parse_args()

    # Resolve parameters
    input_dir = args.input_dir
    output_dir = Path(args.output_root) / f"w{args.window}_s{args.slide}_n{args.n_chunks}"
    window_duration = args.window
    slide = args.slide
    n_chunks = args.n_chunks

    # Logging setup ---------------------------------------------------------
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    dataset_id = f"w{window_duration}_s{slide}_n{n_chunks}"
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{dataset_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
    )

    # Redirect print -> logging (stdout only to preserve tqdm on stderr)
    import sys
    sys.stdout = _LoggerWriter(logging.INFO)

    # Ensure we don't override existing processed folders unless intended
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== AdVitam Batch Feature Extraction for LSTM/CNN+RNN ===")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(
        f"Window duration: {window_duration}s, Slide: {slide}s, Chunks: {n_chunks}"
    )
    if args.failed_filename:
        logging.info(f"Processing only failed files from: {args.failed_filename}")
    else:
        logging.info("Processing all .acq files in input directory")

    # Find .acq files to process
    input_path = Path(input_dir)

    if args.failed_filename:
        # Read failed filenames from the specified file
        try:
            with open(args.failed_filename, "r") as f:
                failed_files_list = [line.strip() for line in f if line.strip()]

            # Convert to full paths
            acq_files = []
            for filename in failed_files_list:
                file_path = input_path / filename
                if file_path.exists():
                    acq_files.append(file_path)
                else:
                    print(f"Warning: Failed file {filename} not found in {input_dir}")

            print(f"Found {len(acq_files)} failed files to reprocess")

        except FileNotFoundError:
            print(f"Error: Failed filename file {args.failed_filename} not found")
            return
        except Exception as e:
            print(f"Error reading failed filename file: {e}")
            return
    else:
        # Process all .acq files
        acq_files = list(input_path.glob("*.acq"))
        print(f"Found {len(acq_files)} .acq files to process")

    if not acq_files:
        print(f"✗ No .acq files found to process")
        return

    print()

    logging.info(f"Starting parallel extraction with {args.workers} worker(s)...")

    successful_files = 0
    failed_files = 0
    failed_filenames = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {
            executor.submit(
                process_single_file,
                str(acq_file),
                str(output_dir),
                window_duration=window_duration,
                slide=slide,
                n_chunks=n_chunks,
            ): acq_file
            for acq_file in sorted(acq_files)
        }

        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Files"):
            acq_file = future_to_file[future]
            try:
                success = future.result()
            except Exception as exc:
                logging.error(f"{acq_file.name} extraction exception: {exc}")
                success = False

            if success:
                successful_files += 1
            else:
                failed_files += 1
                failed_filenames.append(acq_file.name)

    # Save failed filenames to a text file
    failed_files_path = Path(output_dir) / "failed_files.txt"
    if failed_filenames:
        with open(failed_files_path, "w") as f:
            for filename in failed_filenames:
                f.write(f"{filename}\n")
        print(f"\nFailed files saved to: {failed_files_path}")
    else:
        # Create empty file if no failures
        failed_files_path.touch()
        print(f"\nNo failed files - empty list saved to: {failed_files_path}")

    # Create baseline summary if any files were successful
    if successful_files > 0:
        print(f"\n{'='*60}")
        print("CREATING BASELINE FEATURES SUMMARY")
        print(f"{'='*60}")
        create_baseline_summary(output_dir)

    # Summary
    summary_msg = (
        f"BATCH SUMMARY | processed={len(acq_files)} success={successful_files} "
        f"failed={failed_files} success_rate={successful_files/len(acq_files)*100:.1f}% "
        f"out={output_dir} failed_list={failed_files_path}"
    )
    logging.info(summary_msg)

    # Console-friendly final message
    print(summary_msg)


# ---------------------------------------------------------------------------
# Ensure worker processes also pipe their prints to logging
# ---------------------------------------------------------------------------

def _setup_worker_logging():
    """Redirect stdout/stderr inside worker processes to logging."""
    import sys as _sys
    _sys.stdout = _LoggerWriter(logging.INFO)
    _sys.stderr = _LoggerWriter(logging.ERROR)


# Patch process_single_file to activate logging redirection at start
_orig_process_single_file = process_single_file


def process_single_file(*args, **kwargs):  # type: ignore[override]
    _setup_worker_logging()
    return _orig_process_single_file(*args, **kwargs)


if __name__ == "__main__":
    main()
