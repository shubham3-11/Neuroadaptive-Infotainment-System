"""
Feature Processor for KSS Prediction

This module handles feature preprocessing including scaling, NaN handling,
and feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    Feature processor for KSS prediction pipeline
    """

    def __init__(self, config: dict):
        """
        Initialize the feature processor with configuration.

        Parameters:
        -----------
        config : dict
            Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.scaler = None
        self.imputer = None
        self.feature_mask = None

        # Initialize scaler
        scaler_type = config.get("scaler", "standard")
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}. Using StandardScaler.")
            self.scaler = StandardScaler()

        # Initialize imputer
        impute_strategy = config.get("impute_strategy", "mean")
        self.imputer = SimpleImputer(strategy=impute_strategy)

        logger.info(
            f"Initialized FeatureProcessor with {scaler_type} scaler and {impute_strategy} imputation"
        )

    def process_features(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        participant_ids: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        fit: bool = True,
    ) -> np.ndarray:
        """
        Process features including NaN handling, feature selection, and scaling.

        Parameters:
        -----------
        X : np.ndarray
            Features array of shape (n_samples, n_windows, n_features)
        y : np.ndarray, optional
            Labels array for feature selection
        participant_ids : np.ndarray, optional
            Participant IDs for validation
        feature_names : list, optional
            List of feature names
        fit : bool
            Whether to fit the scaler and imputer (True for training, False for prediction)

        Returns:
        --------
        X_processed : np.ndarray
            Processed features array
        """
        try:
            print(f"FeatureProcessor: Processing features with shape: {X.shape}")

            # Reshape for processing (flatten windows)
            original_shape = X.shape
            X_flat = X.reshape(X.shape[0], -1)

            print(f"FeatureProcessor: Flattened shape: {X_flat.shape}")

            # Handle NaN values
            if np.isnan(X_flat).any():
                print("FeatureProcessor: Handling NaN values...")
                if fit:
                    X_flat = self.imputer.fit_transform(X_flat)
                else:
                    X_flat = self.imputer.transform(X_flat)
                print("FeatureProcessor: NaN imputation completed")

            # Feature selection (remove low variance features)
            if fit and y is not None:
                print("FeatureProcessor: Running feature selection...")
                high_nan_threshold = float(self.config.get("high_nan_threshold", 0.5))
                low_var_threshold = float(self.config.get("low_var_threshold", 1e-6))
                print(
                    f"FeatureProcessor: Feature selection thresholds: NaN < {high_nan_threshold}, variance > {low_var_threshold}"
                )
                self._select_features(X_flat, y, feature_names)
            else:
                print(
                    f"FeatureProcessor: Skipping feature selection (fit={fit}, y is None={y is None})"
                )

            if self.feature_mask is not None:
                X_flat = X_flat[:, self.feature_mask]
                print(
                    f"FeatureProcessor: Feature selection applied. New shape: {X_flat.shape}"
                )
            else:
                print(
                    "FeatureProcessor: No feature mask applied (feature_mask is None)"
                )

            # Scale features
            print("FeatureProcessor: Scaling features...")
            if fit:
                X_flat = self.scaler.fit_transform(X_flat)
            else:
                X_flat = self.scaler.transform(X_flat)
            print("FeatureProcessor: Feature scaling completed")

            # Reshape back to original format
            if len(original_shape) == 3:
                n_samples = original_shape[0]
                n_windows = original_shape[1]
                n_features = X_flat.shape[1] // n_windows

                # Check if the reshape is possible
                if X_flat.shape[1] % n_windows != 0:
                    print(
                        f"FeatureProcessor: Cannot reshape {X_flat.shape[1]} features into {n_windows} windows. Returning flat array."
                    )
                    X_processed = X_flat
                else:
                    X_processed = X_flat.reshape(n_samples, n_windows, n_features)
            else:
                X_processed = X_flat

            print(f"FeatureProcessor: Final processed shape: {X_processed.shape}")
            return X_processed

        except Exception as e:
            print(f"FeatureProcessor: Error processing features: {str(e)}")
            raise

    def _select_features(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        """
        Select features based on variance and NaN threshold.

        Parameters:
        -----------
        X : np.ndarray
            Features array (flattened)
        y : np.ndarray
            Labels array
        feature_names : list, optional
            List of feature names (for original features, not flattened)
        """
        try:
            high_nan_threshold = float(self.config.get("high_nan_threshold", 0.5))
            low_var_threshold = float(self.config.get("low_var_threshold", 1e-6))

            # Calculate feature statistics
            feature_vars = np.var(X, axis=0)
            feature_nan_ratios = np.isnan(X).mean(axis=0)

            # Create feature mask
            self.feature_mask = (feature_nan_ratios < high_nan_threshold) & (
                feature_vars > low_var_threshold
            )

            n_selected = self.feature_mask.sum()
            n_total = len(self.feature_mask)

            print(
                f"FeatureProcessor: Feature selection: {n_selected}/{n_total} features selected"
            )

            # Detailed logging of removed features
            removed_features = []
            for i in range(n_total):
                if not self.feature_mask[i]:
                    # For flattened features, we can't directly map to original feature names
                    # So we'll just show the index and statistics
                    nan_ratio = feature_nan_ratios[i]
                    variance = feature_vars[i]

                    if nan_ratio >= high_nan_threshold:
                        reason = (
                            f"high NaN ratio ({nan_ratio:.3f} >= {high_nan_threshold})"
                        )
                    elif variance <= low_var_threshold:
                        reason = f"low variance ({variance:.2e} <= {low_var_threshold})"
                    else:
                        reason = "unknown"

                    removed_features.append(f"flattened_feature_{i} ({reason})")

            if removed_features:
                print(f"FeatureProcessor: Removed {len(removed_features)} features:")
                for feature in removed_features:
                    print(f"FeatureProcessor:   - {feature}")
            else:
                print("FeatureProcessor: No features were removed")

        except Exception as e:
            print(f"FeatureProcessor: Error in feature selection: {str(e)}")
            # If feature selection fails, keep all features
            self.feature_mask = np.ones(X.shape[1], dtype=bool)

    def save_processor(self, filepath: str):
        """
        Save the fitted processor to disk.

        Parameters:
        -----------
        filepath : str
            Path to save the processor
        """
        import joblib

        processor_data = {
            "scaler": self.scaler,
            "imputer": self.imputer,
            "feature_mask": self.feature_mask,
            "config": self.config,
        }
        joblib.dump(processor_data, filepath)
        logger.info(f"Feature processor saved to {filepath}")

    def load_processor(self, filepath: str):
        """
        Load a fitted processor from disk.

        Parameters:
        -----------
        filepath : str
            Path to load the processor from
        """
        import joblib

        processor_data = joblib.load(filepath)
        self.scaler = processor_data["scaler"]
        self.imputer = processor_data["imputer"]
        self.feature_mask = processor_data["feature_mask"]
        self.config = processor_data["config"]
        logger.info(f"Feature processor loaded from {filepath}")
