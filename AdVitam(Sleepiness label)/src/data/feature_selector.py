"""
Feature Selector for AdVitam Dataset

This module handles different feature combinations including baseline features
for the AdVitam dataset.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selector for combining driving and baseline features
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature selector with configuration.

        Parameters:
        -----------
        config : dict
            Configuration dictionary containing feature selection parameters
        """
        self.config = config
        self.feature_selection_config = config.get("feature_selection", {})
        self.mode = self.feature_selection_config.get("mode", "driving_only")
        self.baseline_handling = self.feature_selection_config.get(
            "baseline_handling", {}
        )

        # Initialize scalers
        self.driving_scaler = None
        self.baseline_scaler = None

        logger.info(f"Feature selector initialized with mode: {self.mode}")

    def load_baseline_features(
        self, features_dir: Path, participant_id: str
    ) -> Optional[Dict[str, float]]:
        """
        Load baseline features for a specific participant.

        Parameters:
        -----------
        features_dir : Path
            Directory containing feature files
        participant_id : str
            Participant ID (e.g., "01_AC16")

        Returns:
        --------
        dict or None
            Baseline features dictionary or None if not found
        """
        baseline_file = features_dir / f"{participant_id}_baseline_features.json"

        if not baseline_file.exists():
            logger.warning(f"Baseline features not found for {participant_id}")
            return None

        try:
            with open(baseline_file, "r") as f:
                baseline_features = json.load(f)
            logger.debug(
                f"Loaded {len(baseline_features)} baseline features for {participant_id}"
            )
            return baseline_features
        except Exception as e:
            logger.error(f"Error loading baseline features for {participant_id}: {e}")
            return None

    def get_feature_names(
        self,
        driving_feature_names: List[str],
        baseline_features: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Get feature names based on the selected mode.

        Parameters:
        -----------
        driving_feature_names : list
            List of driving feature names
        baseline_features : dict or None
            Baseline features dictionary

        Returns:
        --------
        list
            List of feature names for the selected mode
        """
        feature_names = []

        if self.mode == "driving_only":
            feature_names = driving_feature_names.copy()

        elif self.mode == "driving_baseline":
            feature_names = driving_feature_names.copy()
            if baseline_features:
                # Add baseline feature names (excluding metadata)
                baseline_names = [
                    name
                    for name in baseline_features.keys()
                    if name.startswith("baseline_")
                    and not name.endswith(("_seconds", "_minutes", "_time"))
                ]
                feature_names.extend(baseline_names)

        elif self.mode == "driving_minus_baseline":
            feature_names = driving_feature_names.copy()

        elif self.mode == "all_combinations":
            feature_names = driving_feature_names.copy()
            if baseline_features:
                # Add baseline feature names
                baseline_names = [
                    name
                    for name in baseline_features.keys()
                    if name.startswith("baseline_")
                    and not name.endswith(("_seconds", "_minutes", "_time"))
                ]
                feature_names.extend(baseline_names)
                # Add difference feature names
                diff_names = [f"diff_{name}" for name in driving_feature_names]
                feature_names.extend(diff_names)

        logger.info(f"Feature mode '{self.mode}': {len(feature_names)} features")
        return feature_names

    def combine_features(
        self,
        driving_features: np.ndarray,
        driving_feature_names: List[str],
        baseline_features: Optional[Dict[str, float]] = None,
        participant_id: str = "",
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Combine driving and baseline features based on the selected mode.

        Parameters:
        -----------
        driving_features : np.ndarray
            Driving features array of shape (n_samples, n_windows, n_features)
        driving_feature_names : list
            List of driving feature names
        baseline_features : dict or None
            Baseline features dictionary
        participant_id : str
            Participant ID for logging

        Returns:
        --------
        tuple : (combined_features, feature_names)
            Combined features array and feature names
        """

        # Define safe_float at the top so it's available in all branches
        def safe_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        n_samples, n_windows, n_driving_features = driving_features.shape

        if self.mode == "driving_only":
            logger.debug(
                f"Mode 'driving_only': using {n_driving_features} driving features for {participant_id}"
            )
            return driving_features, driving_feature_names

        elif self.mode == "driving_baseline":
            if baseline_features is None:
                logger.warning(
                    f"No baseline features available for {participant_id}, falling back to driving_only"
                )
                return driving_features, driving_feature_names

            # Extract baseline feature values (excluding metadata)
            baseline_names = [
                name
                for name in baseline_features.keys()
                if name.startswith("baseline_")
                and not name.endswith(("_seconds", "_minutes", "_time"))
            ]
            baseline_values = [baseline_features[name] for name in baseline_names]

            # Robustly convert to float, replace non-numeric with np.nan, then with 0.0
            baseline_array = np.array(
                [safe_float(v) for v in baseline_values], dtype=float
            )
            baseline_array = np.where(np.isnan(baseline_array), 0.0, baseline_array)

            # Normalize baseline features if requested
            if self.baseline_handling.get("normalize_baseline", True):
                baseline_scaler_type = self.baseline_handling.get(
                    "baseline_scaler", "standard"
                )
                if baseline_scaler_type == "robust":
                    self.baseline_scaler = RobustScaler()
                else:
                    self.baseline_scaler = StandardScaler()
                baseline_array = self.baseline_scaler.fit_transform(
                    baseline_array.reshape(1, -1)
                ).flatten()

            # Repeat baseline features for each sample and window
            baseline_expanded = np.tile(baseline_array, (n_samples, n_windows, 1))

            # Concatenate driving and baseline features
            combined_features = np.concatenate(
                [driving_features, baseline_expanded], axis=2
            )
            combined_names = driving_feature_names + baseline_names

            logger.debug(
                f"Mode 'driving_baseline': {n_driving_features} driving + {len(baseline_names)} baseline = {combined_features.shape[2]} total features for {participant_id}"
            )
            return combined_features, combined_names

        elif self.mode == "driving_minus_baseline":
            if baseline_features is None:
                logger.warning(
                    f"No baseline features available for {participant_id}, falling back to driving_only"
                )
                return driving_features, driving_feature_names

            # Extract baseline feature values for driving features only
            baseline_values = []
            for driving_name in driving_feature_names:
                baseline_name = f"baseline_{driving_name}"
                if baseline_name in baseline_features:
                    baseline_values.append(baseline_features[baseline_name])
                else:
                    baseline_values.append(0.0)  # Default if baseline feature not found

            baseline_array = np.array(
                [safe_float(v) for v in baseline_values], dtype=float
            )
            baseline_array = np.where(np.isnan(baseline_array), 0.0, baseline_array)

            # Normalize baseline features if requested
            if self.baseline_handling.get("normalize_baseline", True):
                baseline_scaler_type = self.baseline_handling.get(
                    "baseline_scaler", "standard"
                )
                if baseline_scaler_type == "robust":
                    self.baseline_scaler = RobustScaler()
                else:
                    self.baseline_scaler = StandardScaler()
                baseline_array = self.baseline_scaler.fit_transform(
                    baseline_array.reshape(1, -1)
                ).flatten()

            # Repeat baseline features for each sample and window
            baseline_expanded = np.tile(baseline_array, (n_samples, n_windows, 1))

            # Calculate difference: driving - baseline
            difference_features = driving_features - baseline_expanded

            logger.debug(
                f"Mode 'driving_minus_baseline': {n_driving_features} difference features for {participant_id}"
            )
            return difference_features, driving_feature_names

        elif self.mode == "all_combinations":
            if baseline_features is None:
                logger.warning(
                    f"No baseline features available for {participant_id}, falling back to driving_only"
                )
                return driving_features, driving_feature_names

            # Extract baseline feature values
            baseline_names = [
                name
                for name in baseline_features.keys()
                if name.startswith("baseline_")
                and not name.endswith(("_seconds", "_minutes", "_time"))
            ]
            baseline_values = [baseline_features[name] for name in baseline_names]

            baseline_array = np.array(
                [safe_float(v) for v in baseline_values], dtype=float
            )
            baseline_array = np.where(np.isnan(baseline_array), 0.0, baseline_array)

            # Normalize baseline features if requested
            if self.baseline_handling.get("normalize_baseline", True):
                baseline_scaler_type = self.baseline_handling.get(
                    "baseline_scaler", "standard"
                )
                if baseline_scaler_type == "robust":
                    self.baseline_scaler = RobustScaler()
                else:
                    self.baseline_scaler = StandardScaler()
                baseline_array = self.baseline_scaler.fit_transform(
                    baseline_array.reshape(1, -1)
                ).flatten()

            # Repeat baseline features for each sample and window
            baseline_expanded = np.tile(baseline_array, (n_samples, n_windows, 1))

            # Calculate difference features
            baseline_driving_values = []
            for driving_name in driving_feature_names:
                baseline_name = f"baseline_{driving_name}"
                if baseline_name in baseline_features:
                    baseline_driving_values.append(baseline_features[baseline_name])
                else:
                    baseline_driving_values.append(0.0)

            baseline_driving_array = np.array(
                [safe_float(v) for v in baseline_driving_values], dtype=float
            )
            baseline_driving_array = np.where(
                np.isnan(baseline_driving_array), 0.0, baseline_driving_array
            )
            baseline_driving_expanded = np.tile(
                baseline_driving_array, (n_samples, n_windows, 1)
            )
            difference_features = driving_features - baseline_driving_expanded

            # Concatenate all features: driving + baseline + difference
            combined_features = np.concatenate(
                [driving_features, baseline_expanded, difference_features], axis=2
            )
            combined_names = (
                driving_feature_names
                + baseline_names
                + [f"diff_{name}" for name in driving_feature_names]
            )

            logger.debug(
                f"Mode 'all_combinations': {n_driving_features} driving + {len(baseline_names)} baseline + {n_driving_features} difference = {combined_features.shape[2]} total features for {participant_id}"
            )
            return combined_features, combined_names

        else:
            logger.warning(f"Unknown mode '{self.mode}', falling back to driving_only")
            return driving_features, driving_feature_names

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the current feature selection configuration.

        Returns:
        --------
        dict
            Feature selection information
        """
        return {
            "mode": self.mode,
            "baseline_handling": self.baseline_handling,
            "description": {
                "driving_only": "Only driving features (original behavior)",
                "driving_baseline": "Driving features + baseline features as additional dimensions",
                "driving_minus_baseline": "Driving features - baseline features (difference)",
                "all_combinations": "All three options combined (driving, baseline, difference)",
            },
        }
