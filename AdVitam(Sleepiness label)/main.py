#!/usr/bin/env python3
"""
KSS Prediction Pipeline - Main Entry Point

This script provides a comprehensive pipeline for predicting Karolinska Sleepiness Scale (KSS)
scores from physiological data using LSTM models.

Usage:
    python main.py --mode train --config configs/advitam.yaml --output results/experiment1
"""

import argparse
import sys
import warnings
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
from models.lstm import LSTMPipeline
from data.advitam_loader import AdVitamDataLoader
from data.feature_processor import FeatureProcessor
from utils.config import Config
from utils.logger import setup_logger
from utils.plotting import KSSPlotter

warnings.filterwarnings("ignore")


def setup_random_state(seed: int = 42):
    """
    Set up random state for reproducible results.

    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    """
    import random
    import torch

    # Set seeds for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🔧 Random seed set to: {seed}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="KSS Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (train + evaluate + predict)
  python main.py --mode all --config configs/default.yaml
  
  # Run training and evaluation
  python main.py --mode train,evaluate --config configs/baseline.yaml

  # Run training and evaluation with plots
  python main.py --mode train,evaluate --config configs/baseline.yaml --plot

  # Run training and evaluation with custom seed
  python main.py --mode train,evaluate --config configs/baseline.yaml --seed 123
  
  # Run training and evaluation with KSS difference mode
  python main.py --mode train,evaluate --prediction_mode kss_difference
  
  # Run prediction only with custom input
  python main.py --mode predict --input data/test_features.npy
  
  # Run prediction with KSS difference mode
  python main.py --mode predict --prediction_mode kss_difference
  
      """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Pipeline modes: 'all', 'train', 'evaluate', 'predict', or comma-separated combinations (e.g., 'train,evaluate')",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input file for prediction mode (optional if running full pipeline)",
    )

    parser.add_argument(
        "--output", type=str, help="Output directory for results (default: results/)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots after training/evaluation",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--prediction_mode",
        type=str,
        choices=["kss_direct", "kss_difference"],
        help="Prediction mode: 'kss_direct' (predict KSS directly) or 'kss_difference' (predict KSS difference from baseline)",
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        config = Config.from_yaml(config_path)
        print(f"DEBUG: Loaded YAML config: {config.to_dict()}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("Creating default configuration...")
        return Config()
    except yaml.YAMLError as e:
        print(f"❌ Error parsing configuration file: {e}")
        return Config()


def filter_data_by_chunks(
    X, y, participant_ids, min_chunks=5, max_chunks=5, logger=None
):
    """
    Filter data to keep only scenarios with at least min_chunks chunks.
    For scenarios with more than max_chunks, keep only the first max_chunks.

    Parameters:
    -----------
    X : np.ndarray
        Features array
    y : np.ndarray
        Labels array
    participant_ids : np.ndarray
        Participant IDs array
    min_chunks : int
        Minimum number of chunks required
    max_chunks : int
        Maximum number of chunks to keep
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    X_filtered, y_filtered, participant_ids_filtered : tuple
        Filtered arrays
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA FILTERING BY CHUNKS")
    logger.info("=" * 60)

    # Group data by participant
    unique_participants = np.unique(participant_ids)
    logger.info(f"Total participants before filtering: {len(unique_participants)}")

    X_filtered = []
    y_filtered = []
    participant_ids_filtered = []

    kept_participants = 0
    discarded_participants = 0

    for participant in unique_participants:
        # Get data for this participant
        mask = participant_ids == participant
        participant_X = X[mask]
        participant_y = y[mask]

        n_chunks = len(participant_y)

        if n_chunks < min_chunks:
            logger.info(
                f"❌ Participant {participant}: {n_chunks} chunks (discarded - < {min_chunks})"
            )
            discarded_participants += 1
            continue
        elif n_chunks > max_chunks:
            logger.info(
                f"✂️  Participant {participant}: {n_chunks} chunks (keeping first {max_chunks})"
            )
            participant_X = participant_X[:max_chunks]
            participant_y = participant_y[:max_chunks]
            kept_participants += 1
        else:
            logger.info(f"✅ Participant {participant}: {n_chunks} chunks (kept as is)")
            kept_participants += 1

        X_filtered.append(participant_X)
        y_filtered.extend(participant_y)
        participant_ids_filtered.extend([participant] * len(participant_y))

    X_filtered = np.vstack(X_filtered)
    y_filtered = np.array(y_filtered)
    participant_ids_filtered = np.array(participant_ids_filtered)

    logger.info(f"📊 FILTERING SUMMARY:")
    logger.info(f"   - Kept scenarios: {kept_participants}")
    logger.info(f"   - Discarded scenarios: {discarded_participants}")
    logger.info(f"   - Total samples after filtering: {len(X_filtered)}")
    logger.info(f"   - Data shape: {X_filtered.shape}")

    return X_filtered, y_filtered, participant_ids_filtered


def handle_missing_values(X, y, participant_ids, feature_names=None, logger=None):
    """
    Handle missing values in the data using rolling mean approach.

    Parameters:
    -----------
    X : np.ndarray
        Features array
    y : np.ndarray
        Labels array
    participant_ids : np.ndarray
        Participant IDs array
    feature_names : list, optional
        List of feature names
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    X_cleaned, y_cleaned, participant_ids_cleaned : tuple
        Cleaned arrays
    """
    logger.info("=" * 60)
    logger.info("STEP 2: MISSING VALUE HANDLING")
    logger.info("=" * 60)

    original_shape = X.shape
    logger.info(f"Original data shape: {original_shape}")

    # Check for NaN in features
    nan_features = np.isnan(X).any(
        axis=(0, 1)
    )  # Check each feature across all samples and windows
    n_nan_features = nan_features.sum()
    logger.info(f"Features with NaN values: {n_nan_features}/{X.shape[2]}")

    # Check for NaN in labels
    nan_labels = np.isnan(y)
    n_nan_labels = nan_labels.sum()
    logger.info(f"Labels with NaN values: {n_nan_labels}/{len(y)}")

    # Remove samples with NaN labels
    if n_nan_labels > 0:
        logger.info(f"Removing {n_nan_labels} samples with NaN labels")
        valid_mask = ~nan_labels
        X = X[valid_mask]
        y = y[valid_mask]
        participant_ids = participant_ids[valid_mask]

    # Remove features with all missing values
    if n_nan_features > 0:
        # Check which features have all NaN values
        all_nan_features = np.isnan(X).all(axis=(0, 1))
        n_all_nan = all_nan_features.sum()

        if n_all_nan > 0:
            logger.info(f"Removing {n_all_nan} features with all missing values")
            if feature_names:
                removed_features = [
                    feature_names[i]
                    for i in range(len(feature_names))
                    if all_nan_features[i]
                ]
                logger.info(f"Removed features: {removed_features}")
            valid_features = ~all_nan_features
            X = X[:, :, valid_features]
            if feature_names:
                feature_names = [
                    feature_names[i]
                    for i in range(len(feature_names))
                    if valid_features[i]
                ]

        # Fill remaining NaN values using rolling mean approach
        logger.info("Filling remaining NaN values using rolling mean approach")
        for i in range(X.shape[2]):  # For each feature
            feature_data = X[:, :, i]
            if np.isnan(feature_data).any():
                feature_name = feature_names[i] if feature_names else f"feature_{i}"

                # Count how many samples have NaN values for this feature
                nan_samples = np.isnan(feature_data).any(axis=1)  # Check each sample
                n_nan_samples = nan_samples.sum()
                total_samples = feature_data.shape[0]

                logger.info(
                    f"   - Processing feature {feature_name} with NaN values ({n_nan_samples}/{total_samples} samples)"
                )

                # For each sample (chunk)
                for sample_idx in range(feature_data.shape[0]):
                    chunk_data = feature_data[
                        sample_idx, :
                    ]  # Get the chunk for this sample

                    if np.isnan(chunk_data).any():
                        # Find NaN positions in this chunk
                        nan_positions = np.where(np.isnan(chunk_data))[0]

                        for nan_pos in nan_positions:
                            # Try rolling mean with window size 3 (current, previous, next)
                            window_size = 3
                            start_idx = max(0, nan_pos - window_size // 2)
                            end_idx = min(
                                len(chunk_data), nan_pos + window_size // 2 + 1
                            )

                            # Get nearby values (excluding the current NaN position)
                            nearby_values = []
                            for j in range(start_idx, end_idx):
                                if j != nan_pos and not np.isnan(chunk_data[j]):
                                    nearby_values.append(chunk_data[j])

                            if len(nearby_values) > 0:
                                # Use rolling mean if nearby values are available
                                chunk_data[nan_pos] = np.mean(nearby_values)
                                logger.debug(
                                    f"  Sample {sample_idx}, position {nan_pos}: filled with rolling mean ({len(nearby_values)} nearby values)"
                                )
                            else:
                                # Use chunk mean if no nearby values are available
                                chunk_mean = np.nanmean(chunk_data)
                                chunk_data[nan_pos] = chunk_mean
                                logger.debug(
                                    f"  Sample {sample_idx}, position {nan_pos}: filled with chunk mean (no nearby values)"
                                )

                        # Update the feature data
                        feature_data[sample_idx, :] = chunk_data

                # Update the feature in the main array
                X[:, :, i] = feature_data

    logger.info(f"Final data shape: {X.shape}")
    logger.info(f"✅ Missing value handling completed")

    return X, y, participant_ids


def run_training(
    data_loader,
    feature_processor,
    model_pipeline,
    output_dir,
    logger,
    prediction_mode="kss_direct",
):
    """Run the training pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load and process data
    X, y, participant_ids, feature_names = data_loader.load_features_and_labels()
    logger.info(
        f"Loaded data: X={X.shape}, y={y.shape}, participants={len(np.unique(participant_ids))}"
    )

    # Handle prediction mode for training
    if prediction_mode == "kss_difference":
        logger.info(
            "Training mode: kss_difference - Converting labels to differences from baseline"
        )
        baseline_kss = data_loader.get_baseline_kss_values()

        if baseline_kss:
            # Convert labels to differences from baseline
            y_differences = np.zeros_like(y)
            for i, participant_id in enumerate(participant_ids):
                if participant_id in baseline_kss:
                    baseline_value = baseline_kss[participant_id]
                    y_differences[i] = y[i] - baseline_value
                else:
                    logger.warning(
                        f"No baseline KSS found for {participant_id}, using original label"
                    )
                    y_differences[i] = y[i]

            y = y_differences
            logger.info(
                f"Converted labels to differences. New y range: {y.min():.2f} - {y.max():.2f}"
            )
        else:
            logger.warning("No baseline KSS values found, using original labels")
    else:
        logger.info(f"Training mode: {prediction_mode} - Using original labels")

    # Process features
    X_processed = feature_processor.process_features(X)
    logger.info(f"Processed features: {X_processed.shape}")

    # Train model
    model, history, metrics, data_splits = model_pipeline.train(
        X_processed, y, participant_ids, logger
    )

    # Save results
    model_pipeline.save_results(metrics, history, output_dir)
    model_pipeline.save_model(str(output_dir / "model.pth"))

    logger.info(f"✅ Training completed! Results saved to {output_dir}")

    return X_processed, y, participant_ids, metrics, data_splits


def run_evaluation(
    data_loader,
    feature_processor,
    model_pipeline,
    output_dir,
    X_processed=None,
    y=None,
    participant_ids=None,
    logger=None,
    prediction_mode="kss_direct",
):
    """Run the evaluation pipeline with detailed test results."""
    logger.info("=" * 60)
    logger.info("STEP 4: MODEL EVALUATION")
    logger.info("=" * 60)

    # Load trained model
    model_path = output_dir / "model.pth"
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please run training first or provide a valid model path.")
        return None

    # Load and preprocess data if not provided
    if X_processed is None or y is None or participant_ids is None:
        logger.info("Loading data for evaluation...")
        X, y, participant_ids, feature_names = data_loader.load_features_and_labels()
        if X is None or y is None:
            logger.error("Failed to load data for evaluation.")
            return None

        # Apply same preprocessing as training
        X, y, participant_ids = filter_data_by_chunks(
            X, y, participant_ids, logger=logger
        )
        X, y, participant_ids = handle_missing_values(
            X, y, participant_ids, logger=logger
        )
        X_processed = X

    # Handle prediction mode for evaluation
    original_y = y.copy()  # Keep original for final evaluation
    if prediction_mode == "kss_difference":
        logger.info(
            "Evaluation mode: kss_difference - Converting labels to differences from baseline"
        )
        baseline_kss = data_loader.get_baseline_kss_values()

        if baseline_kss:
            # Convert labels to differences from baseline
            y_differences = np.zeros_like(y)
            for i, participant_id in enumerate(participant_ids):
                if participant_id in baseline_kss:
                    baseline_value = baseline_kss[participant_id]
                    y_differences[i] = y[i] - baseline_value
                else:
                    logger.warning(
                        f"No baseline KSS found for {participant_id}, using original label"
                    )
                    y_differences[i] = y[i]

            y = y_differences
            logger.info(
                f"Converted labels to differences. New y range: {y.min():.2f} - {y.max():.2f}"
            )
        else:
            logger.warning("No baseline KSS values found, using original labels")
    else:
        logger.info(f"Evaluation mode: {prediction_mode} - Using original labels")

    # Evaluate model
    metrics = model_pipeline.evaluate(
        str(model_path), X_processed, y, participant_ids, logger=logger
    )

    # Save evaluation results
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    model_pipeline.save_evaluation_results(metrics, eval_dir)

    logger.info(f"✅ Evaluation completed. Results saved to {eval_dir}")
    return metrics


def print_summary_table(train_metrics, val_metrics, test_metrics, logger):
    """Print a summary table with key metrics."""
    logger.info("=" * 60)
    logger.info("STEP 5: MODEL SUMMARY")
    logger.info("=" * 60)

    # Create summary table
    summary_data = {
        "Metric": ["MSE", "MAE", "RMSE", "R²"],
        "Train": [
            f"{train_metrics.get('mse', 'N/A'):.4f}",
            f"{train_metrics.get('mae', 'N/A'):.4f}",
            f"{train_metrics.get('rmse', 'N/A'):.4f}",
            f"{train_metrics.get('r2', 'N/A'):.4f}",
        ],
        "Validation": [
            f"{val_metrics.get('mse', 'N/A'):.4f}",
            f"{val_metrics.get('mae', 'N/A'):.4f}",
            f"{val_metrics.get('rmse', 'N/A'):.4f}",
            f"{val_metrics.get('r2', 'N/A'):.4f}",
        ],
        "Test": [
            f"{test_metrics.get('mse', 'N/A'):.4f}",
            f"{test_metrics.get('mae', 'N/A'):.4f}",
            f"{test_metrics.get('rmse', 'N/A'):.4f}",
            f"{test_metrics.get('r2', 'N/A'):.4f}",
        ],
    }

    df = pd.DataFrame(summary_data)

    logger.info("📊 MODEL PERFORMANCE SUMMARY")
    logger.info("-" * 60)

    # Print table with better formatting
    logger.info(f"{'Metric':<8} {'Train':<12} {'Validation':<12} {'Test':<12}")
    logger.info("-" * 60)
    for _, row in df.iterrows():
        logger.info(
            f"{row['Metric']:<8} {row['Train']:<12} {row['Validation']:<12} {row['Test']:<12}"
        )

    # Add interpretation
    logger.info("📈 PERFORMANCE INTERPRETATION:")
    logger.info("-" * 60)
    test_mse = test_metrics.get("mse", 0)
    test_mae = test_metrics.get("mae", 0)
    test_rmse = test_metrics.get("rmse", 0)
    test_r2 = test_metrics.get("r2", 0)

    if test_r2 > 0.7:
        logger.info("✅ Excellent performance (R² > 0.7)")
    elif test_r2 > 0.5:
        logger.info("✅ Good performance (R² > 0.5)")
    elif test_r2 > 0.3:
        logger.info("⚠️  Moderate performance (R² > 0.3)")
    elif test_r2 > 0:
        logger.info("⚠️  Poor performance (R² > 0)")
    else:
        logger.info(
            "❌ Very poor performance (R² < 0) - Model performs worse than baseline"
        )

    logger.info(f"   - Test MSE: {test_mse:.4f}")
    logger.info(f"   - Test MAE: {test_mae:.4f}")
    logger.info(f"   - Test RMSE: {test_rmse:.4f}")
    logger.info(f"   - Test R²: {test_r2:.4f}")


def run_plotting(
    output_dir, logger=None, training_data=None, model_pipeline=None, data_splits=None
):
    """Generate plots from training results."""
    logger.info("=" * 60)
    logger.info("STARTING PLOTTING PIPELINE")
    logger.info("=" * 60)

    # Create plotter
    plotter = KSSPlotter(output_dir)

    try:
        if training_data is not None and model_pipeline is not None:
            # Use actual training data for better plots
            X_processed, y, participant_ids, training_metrics = training_data

            # Load history from results
            history_file = output_dir / "history.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                logger.warning("History file not found, using empty history")
                history = {}

            # Generate predictions for plotting
            model_path = output_dir / "model.pth"
            if model_path.exists():
                # Extract train/val/test data for distribution plot
                y_train = data_splits.get("y_train") if data_splits else None
                y_val = data_splits.get("y_val") if data_splits else None
                y_test = data_splits.get("y_test") if data_splits else None

                # Generate test set predictions for proper evaluation
                if data_splits:
                    # Get test indices from the original data
                    test_idx = model_pipeline.get_test_indices(
                        X_processed, y, participant_ids
                    )
                    X_test = X_processed[test_idx]
                    y_test_true = y[test_idx]

                    # Generate predictions only for test set
                    y_test_pred = model_pipeline.predict(str(model_path), X_test)

                    # Use test set for predictions vs actual plot
                    y_pred_for_plot = y_test_pred
                    y_true_for_plot = y_test_true
                else:
                    # Fallback to all data if splits not available
                    y_pred_for_plot = model_pipeline.predict(
                        str(model_path), X_processed
                    )
                    y_true_for_plot = y

                # Generate all plots with real data
                plotter.plot_all_results(
                    history,
                    y_true_for_plot,
                    y_pred_for_plot,
                    training_metrics,
                    y_train,
                    y_val,
                    y_test,
                )
            else:
                logger.warning("Model file not found, using basic plots")
                plotter.load_and_plot_from_results(output_dir)
        else:
            # Generate basic plots from saved results
            plotter.load_and_plot_from_results(output_dir)

        logger.info(f"✅ Plots generated successfully in: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"❌ Error generating plots: {e}")
        return False


def run_prediction(
    model_pipeline,
    feature_processor,
    output_dir,
    input_file=None,
    logger=None,
    prediction_mode="kss_direct",
    data_loader=None,
):
    """Run the prediction pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING PREDICTION PIPELINE")
    logger.info("=" * 60)

    # Load model
    model_path = output_dir / "model.pth"
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please run training first or provide a valid model path.")
        return None

    # Load input data
    if input_file is None:
        logger.info("No input file provided. Using test data for demonstration...")
        # Create dummy data for demonstration
        X = np.random.randn(10, 5, 20)  # 10 samples, 5 windows, 20 features
        logger.info(f"Using dummy data with shape: {X.shape}")
    else:
        logger.info(f"Loading input data from: {input_file}")
        X = np.load(input_file)

    # Process features
    X_processed = X

    # Make predictions
    predictions = model_pipeline.predict(str(model_path), X_processed)

    # Handle prediction mode
    if prediction_mode == "kss_difference" and data_loader is not None:
        logger.info(
            "Mode: kss_difference - Adding baseline KSS values back to predictions"
        )

        # Get baseline KSS values
        baseline_kss = data_loader.get_baseline_kss_values()

        if baseline_kss:
            # For demonstration, we'll use the first baseline value
            # In practice, you'd need to map predictions to specific participants
            first_baseline = list(baseline_kss.values())[0]
            logger.info(f"Using baseline KSS value: {first_baseline}")

            # Add baseline back to predictions
            final_predictions = predictions + first_baseline
            logger.info(
                f"Predictions after adding baseline: {final_predictions.min():.2f} - {final_predictions.max():.2f}"
            )
        else:
            logger.warning("No baseline KSS values found, using direct predictions")
            final_predictions = predictions
    else:
        logger.info(f"Mode: {prediction_mode} - Using direct predictions")
        final_predictions = predictions

    # Save predictions
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    np.save(pred_dir / "predictions.npy", final_predictions)

    # Save as CSV for easier viewing
    pred_df = pd.DataFrame(
        {
            "sample_id": range(len(final_predictions)),
            "predicted_kss": final_predictions.flatten(),
            "prediction_mode": prediction_mode,
        }
    )

    if prediction_mode == "kss_difference" and data_loader is not None:
        baseline_kss = data_loader.get_baseline_kss_values()
        if baseline_kss:
            first_baseline = list(baseline_kss.values())[0]
            pred_df["baseline_kss"] = first_baseline
            pred_df["predicted_difference"] = predictions.flatten()
            pred_df["final_prediction"] = final_predictions.flatten()

    pred_df.to_csv(pred_dir / "predictions.csv", index=False)

    logger.info(f"✅ Predictions saved to {pred_dir}")
    logger.info(f"Prediction shape: {final_predictions.shape}")
    logger.info(
        f"Prediction range: {final_predictions.min():.2f} - {final_predictions.max():.2f}"
    )

    return final_predictions


def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("KSS PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Parse arguments
    args = parse_arguments()
    modes = [mode.strip() for mode in args.mode.split(",")]

    # Setup random state for reproducibility
    setup_random_state(args.seed)

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(level=log_level)
    logger.info(f"Pipeline modes: {modes}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output}")

    # Load configuration
    config = load_config(args.config)
    logger.info("Initializing pipeline components...")

    # Initialize components
    data_loader = AdVitamDataLoader(config.data)
    feature_processor = FeatureProcessor(config.preprocessing)
    model_pipeline = LSTMPipeline(config.model)

    # Debug: Log the actual config values being used
    logger.info("=" * 60)
    logger.info("CONFIGURATION DEBUG")
    logger.info("=" * 60)
    logger.info(f"Model config: {config.model}")
    logger.info(
        f"Learning rate from config: {config.model.get('learning_rate', 'NOT_FOUND')}"
    )
    logger.info(
        f"Batch size from config: {config.model.get('batch_size', 'NOT_FOUND')}"
    )
    logger.info(
        f"Num epochs from config: {config.model.get('num_epochs', 'NOT_FOUND')}"
    )
    logger.info("=" * 60)

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = (
            Path("results") / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # NEW: ensure output directory exists and save the configuration used
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        config_save_path = output_dir / "config.yaml"
        config.to_yaml(str(config_save_path))
        logger.info(f"Configuration saved to {config_save_path}")
    except Exception as e:
        logger.warning(f"Failed to save configuration file: {e}")

    # Execute pipeline modes
    training_metrics = None
    evaluation_metrics = None
    prediction_results = None
    plotting_results = None

    try:
        # Determine prediction mode
        prediction_mode = args.prediction_mode
        if prediction_mode is None:
            # Use config default if not specified in command line
            prediction_mode = config.prediction.get("mode", "kss_direct")

        logger.info(f"Using prediction mode: {prediction_mode}")

        if "train" in modes:
            X_processed, y, participant_ids, training_metrics, data_splits = (
                run_training(
                    data_loader,
                    feature_processor,
                    model_pipeline,
                    output_dir,
                    logger,
                    prediction_mode,
                )
            )

        if "evaluate" in modes:
            evaluation_metrics = run_evaluation(
                data_loader,
                feature_processor,
                model_pipeline,
                output_dir,
                X_processed if "train" in modes else None,
                y if "train" in modes else None,
                participant_ids if "train" in modes else None,
                logger,
                prediction_mode,
            )

        if "predict" in modes:
            prediction_results = run_prediction(
                model_pipeline,
                feature_processor,
                output_dir,
                args.input,
                logger,
                prediction_mode,
                data_loader,
            )

        # Generate plots if requested
        if args.plot:
            # Pass training data if available for better plots
            training_data = None
            if training_metrics and "X_processed" in locals():
                training_data = (X_processed, y, participant_ids, training_metrics)
            plotting_results = run_plotting(
                output_dir, logger, training_data, model_pipeline, data_splits
            )

        # Print summary table if we have metrics
        train_metrics = None
        val_metrics = None
        test_metrics = None

        if training_metrics:
            # Extract train metrics from training results
            train_metrics = {
                "mse": training_metrics.get("train_mse", training_metrics.get("mse")),
                "mae": training_metrics.get("train_mae", training_metrics.get("mae")),
                "rmse": training_metrics.get(
                    "train_rmse", training_metrics.get("rmse")
                ),
                "r2": training_metrics.get("train_r2", training_metrics.get("r2")),
            }

            # Extract validation metrics from training results
            val_metrics = {
                "mse": training_metrics.get("val_mse", training_metrics.get("mse")),
                "mae": training_metrics.get("val_mae", training_metrics.get("mae")),
                "rmse": training_metrics.get("val_rmse", training_metrics.get("rmse")),
                "r2": training_metrics.get("val_r2", training_metrics.get("r2")),
            }

            # Extract test metrics from training results
            test_metrics = {
                "mse": training_metrics.get("test_mse", training_metrics.get("mse")),
                "mae": training_metrics.get("test_mae", training_metrics.get("mae")),
                "rmse": training_metrics.get("test_rmse", training_metrics.get("rmse")),
                "r2": training_metrics.get("test_r2", training_metrics.get("r2")),
            }

        # Print summary table if we have training metrics
        if train_metrics:
            print_summary_table(train_metrics, val_metrics, test_metrics, logger)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(f"Traceback: {e.__traceback__}")
        return 1

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Training: {'Completed' if training_metrics else 'Skipped'}")
    logger.info(f"✅ Evaluation: {'Completed' if evaluation_metrics else 'Skipped'}")
    logger.info(f"✅ Prediction: {'Completed' if prediction_results else 'Skipped'}")
    logger.info(f"✅ Plotting: {'Completed' if plotting_results else 'Skipped'}")
    logger.info(f"📁 All results saved to: {output_dir}")
    logger.info("Pipeline completed successfully!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
