"""
AdVitam Data Loader

This module handles loading and preprocessing of features and labels
for the AdVitam dataset with the specific file structure.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import glob
import logging

# Import feature selector
from .feature_selector import FeatureSelector

# Setup logger
logger = logging.getLogger(__name__)


class AdVitamDataLoader:
    """
    Data loader specifically for AdVitam dataset
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader with configuration.

        Parameters:
        -----------
        config : dict
            Configuration dictionary containing data paths and parameters
        """
        self.config = config
        self.features_dir = Path(
            config.get("features_dir", "data/AdVitam/Preprocessed2/Feature")
        )
        self.labels_dir = Path(
            config.get("labels_dir", "data/AdVitam/Preprocessed2/Target")
        )
        self.labels_file = config.get("labels_file", "all_kss_targets.json")
        self.num_participants = config.get("num_participants", None)

        # Initialize feature selector
        self.feature_selector = FeatureSelector(config)

        # Define file combination rules for specific participants
        self.file_combination_rules = {
            "02_DU16": {
                "scenario1": [1, 2],  # Combine files 1 and 2 for scenario1
                "scenario2": [3],  # Use file 3 for scenario2
            }
            # Add more participants as needed:
            # "participant_id": {
            #     "scenario1": [file_numbers],
            #     "scenario2": [file_numbers]
            # }
        }

        logger.info(f"Features directory: {self.features_dir}")
        logger.info(f"Labels directory: {self.labels_dir}")
        logger.info(f"Labels file: {self.labels_file}")
        logger.info(f"Feature selection mode: {self.feature_selector.mode}")
        if self.num_participants is not None:
            logger.info(f"Limiting to first {self.num_participants} participants")
        else:
            logger.info("Using all available participants")

    def load_features_and_labels(
        self,
    ) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[list]
    ]:
        """
        Load features and labels from the AdVitam preprocessed data.

        Returns:
        --------
        X : np.ndarray or None
            Features array of shape (n_samples, n_windows, n_features)
        y : np.ndarray or None
            Labels array of shape (n_samples,)
        participant_ids : np.ndarray or None
            Participant IDs array of shape (n_samples,)
        feature_names : list or None
            List of feature names
        """
        try:
            logger.info("Loading AdVitam features and labels...")

            # Check if directories exist
            if not self.features_dir.exists():
                logger.error(f"Features directory not found: {self.features_dir}")
                return None, None, None, None

            labels_path = self.labels_dir / self.labels_file
            if not labels_path.exists():
                logger.error(f"Labels file not found: {labels_path}")
                return None, None, None, None

            # Load feature names from the first available feature names file
            driving_feature_names = None
            feature_name_files = list(self.features_dir.glob("*_feature_names.txt"))
            if feature_name_files:
                feature_name_file = feature_name_files[0]  # Use the first one
                with open(feature_name_file, "r") as f:
                    driving_feature_names = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                logger.info(
                    f"Loaded {len(driving_feature_names)} driving feature names from {feature_name_file.name}"
                )
            else:
                logger.warning("No feature names file found")

            # Load labels
            logger.info(f"Loading labels from: {labels_path}")
            with open(labels_path, "r") as f:
                all_labels = json.load(f)

            # Ensure all_labels is a dictionary
            if not isinstance(all_labels, dict):
                logger.error(f"Expected dictionary for labels, got {type(all_labels)}")
                return None, None, None, None

            logger.info(f"Found {len(all_labels)} label entries")

            # Limit participants if specified
            if self.num_participants is not None:
                # Get unique participant IDs (without scenario suffix)
                unique_participants = set()
                for label_key in all_labels.keys():
                    participant_id = label_key.replace("_scenario1", "").replace(
                        "_scenario2", ""
                    )
                    unique_participants.add(participant_id)

                # Sort participants to ensure consistent ordering
                sorted_participants = sorted(list(unique_participants))
                limited_participants = sorted_participants[: self.num_participants]

                # Filter labels to only include the limited participants
                filtered_labels = {}
                for label_key, label_data in all_labels.items():
                    participant_id = label_key.replace("_scenario1", "").replace(
                        "_scenario2", ""
                    )
                    if participant_id in limited_participants:
                        filtered_labels[label_key] = label_data

                all_labels = filtered_labels
                logger.info(
                    f"Limited to {len(limited_participants)} participants: {limited_participants}"
                )

            all_features = []
            all_labels_list = []
            all_participants = []

            # Process each label entry
            for label_key, label_data in all_labels.items():
                # Extract participant ID and scenario from key (e.g., "63_AC10_scenario1" -> "63_AC10", "scenario1")
                participant_id = label_key.replace("_scenario1", "").replace(
                    "_scenario2", ""
                )
                scenario = "scenario1" if "_scenario1" in label_key else "scenario2"
                scenario_num = "1" if "_scenario1" in label_key else "2"

                logger.debug(f"Processing participant {participant_id} in {scenario}")

                # Load baseline features for this participant
                baseline_features = self.feature_selector.load_baseline_features(
                    self.features_dir, participant_id
                )

                # Find all feature files for this participant
                all_participant_files = list(
                    self.features_dir.glob(f"{participant_id}_windowed_features_*.npy")
                )
                all_participant_files.sort()  # Sort to ensure consistent ordering

                if not all_participant_files:
                    logger.warning(
                        f"No feature files found for participant {participant_id}"
                    )
                    continue

                logger.debug(
                    f"Found {len(all_participant_files)} files for {participant_id}: {[f.name for f in all_participant_files]}"
                )

                # Determine which files to use based on scenario
                if participant_id in self.file_combination_rules:
                    # Use specific rules for this participant
                    if scenario in self.file_combination_rules[participant_id]:
                        file_numbers = self.file_combination_rules[participant_id][
                            scenario
                        ]
                        feature_files = []
                        for file_num in file_numbers:
                            file_pattern = (
                                f"{participant_id}_windowed_features_{file_num}.npy"
                            )
                            matching_files = list(self.features_dir.glob(file_pattern))
                            if matching_files:
                                feature_files.extend(matching_files)
                            else:
                                logger.warning(f"File not found: {file_pattern}")

                        if not feature_files:
                            logger.warning(
                                f"No files found for {label_key} using rule: {file_numbers}"
                            )
                            continue

                        logger.debug(
                            f"Using rule-based files for {label_key}: {[f.name for f in feature_files]}"
                        )
                    else:
                        logger.warning(f"No rule defined for {label_key}")
                        continue
                else:
                    # Use default logic for participants without specific rules
                    if scenario == "scenario1":
                        # For scenario1, combine first two files (if available)
                        if len(all_participant_files) >= 2:
                            feature_files = all_participant_files[:2]
                            logger.debug(
                                f"Combining first 2 files for {label_key}: {[f.name for f in feature_files]}"
                            )
                        else:
                            feature_files = all_participant_files[:1]
                            logger.debug(
                                f"Using single file for {label_key}: {[f.name for f in feature_files]}"
                            )
                    else:  # scenario2
                        # For scenario2, use the remaining files (if any)
                        if len(all_participant_files) >= 3:
                            feature_files = all_participant_files[2:]
                            logger.debug(
                                f"Using remaining files for {label_key}: {[f.name for f in feature_files]}"
                            )
                        elif len(all_participant_files) == 2:
                            feature_files = all_participant_files[1:2]
                            logger.debug(
                                f"Using second file for {label_key}: {[f.name for f in feature_files]}"
                            )
                        else:
                            logger.warning(f"No files available for {label_key}")
                            continue

                # Load and combine features from selected files
                combined_driving_features = []
                for feature_file in feature_files:
                    features = np.load(feature_file)
                    combined_driving_features.append(features)
                    logger.debug(f"Loaded {features.shape} from {feature_file.name}")

                if not combined_driving_features:
                    logger.warning(f"No features loaded for {label_key}")
                    continue

                # Combine driving features from selected files
                if len(combined_driving_features) > 1:
                    driving_features = np.vstack(combined_driving_features)
                    logger.debug(
                        f"Combined driving features shape: {driving_features.shape}"
                    )
                else:
                    driving_features = combined_driving_features[0]

                # Apply feature selection to combine driving and baseline features
                combined_features, final_feature_names = (
                    self.feature_selector.combine_features(
                        driving_features=driving_features,
                        driving_feature_names=driving_feature_names,
                        baseline_features=baseline_features,
                        participant_id=participant_id,
                    )
                )

                # Get labels for this participant-scenario
                if "chunks" in label_data:
                    labels = label_data["chunks"]

                    # Ensure we have the same number of samples
                    n_samples = min(len(combined_features), len(labels))

                    if n_samples > 0:
                        all_features.append(combined_features[:n_samples])
                        all_labels_list.extend(labels[:n_samples])
                        # Use the full label_key as participant_id to distinguish scenarios
                        all_participants.extend([label_key] * n_samples)

                        logger.debug(f"Added {n_samples} samples for {label_key}")
                    else:
                        logger.warning(f"No valid samples for {label_key}")
                else:
                    logger.warning(f"No chunks found in labels for {label_key}")

            if not all_features:
                logger.error("No features loaded from any participant")
                return None, None, None, None

            # Combine all features
            X = np.vstack(all_features)
            y = np.array(all_labels_list)
            participant_ids = np.array(all_participants)

            # Get final feature names (use the last one since they should be consistent)
            final_feature_names = self.feature_selector.get_feature_names(
                driving_feature_names, None
            )

            logger.info(f"Successfully loaded data:")
            logger.info(f"  - Features shape: {X.shape}")
            logger.info(f"  - Labels shape: {y.shape}")
            logger.info(f"  - Labels range: {y.min():.2f} - {y.max():.2f}")
            logger.info(
                f"  - Number of participants: {len(np.unique(participant_ids))}"
            )
            logger.info(
                f"  - Unique participants: {sorted(np.unique(participant_ids))}"
            )
            logger.info(f"  - Feature selection mode: {self.feature_selector.mode}")
            logger.info(f"  - Final feature count: {len(final_feature_names)}")

            return X, y, participant_ids, final_feature_names

        except Exception as e:
            logger.error(f"Error loading AdVitam features and labels: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None, None, None

    def load_features_only(self) -> Optional[np.ndarray]:
        """
        Load only features (for prediction mode).

        Returns:
        --------
        X : np.ndarray or None
            Features array
        """
        try:
            logger.info("Loading AdVitam features only...")

            if not self.features_dir.exists():
                logger.error(f"Features directory not found: {self.features_dir}")
                return None

            # Load all feature files
            feature_files = list(self.features_dir.glob("*.npy"))

            if not feature_files:
                logger.error("No feature files found")
                return None

            all_features = []
            for feature_file in feature_files:
                features = np.load(feature_file)
                all_features.append(features)
                logger.debug(f"Loaded {features.shape} from {feature_file.name}")

            if not all_features:
                logger.error("No features loaded")
                return None

            X = np.vstack(all_features)
            logger.info(f"Loaded {len(X)} samples with shape {X.shape}")

            return X

        except Exception as e:
            logger.error(f"Error loading AdVitam features: {str(e)}")
            return None

    def validate_data(self) -> bool:
        """
        Validate that the data can be loaded successfully.

        Returns:
        --------
        bool
            True if data is valid, False otherwise
        """
        try:
            X, y, participant_ids, feature_names = self.load_features_and_labels()
            if (
                X is None
                or y is None
                or participant_ids is None
                or feature_names is None
            ):
                return False

            # Basic validation
            if len(X) != len(y) or len(X) != len(participant_ids):
                logger.error("Mismatch in data lengths")
                return False

            if np.isnan(X).any():
                logger.warning("Features contain NaN values")

            if np.isnan(y).any():
                logger.warning("Labels contain NaN values")

            logger.info("Data validation passed")
            return True

        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the current feature selection configuration.

        Returns:
        --------
        dict
            Feature selection information
        """
        return self.feature_selector.get_feature_info()

    def get_baseline_kss_values(self) -> Dict[str, float]:
        """
        Get baseline KSS values for all participants.

        Returns:
        --------
        dict
            Dictionary mapping participant_scenario to baseline KSS value
        """
        try:
            labels_path = self.labels_dir / self.labels_file
            if not labels_path.exists():
                logger.error(f"Labels file not found: {labels_path}")
                return {}

            with open(labels_path, "r") as f:
                all_labels = json.load(f)

            baseline_kss = {}
            for label_key, label_data in all_labels.items():
                if "KSS_0" in label_data:
                    baseline_kss[label_key] = label_data["KSS_0"]

            logger.info(
                f"Loaded baseline KSS values for {len(baseline_kss)} participants"
            )
            return baseline_kss

        except Exception as e:
            logger.error(f"Error loading baseline KSS values: {str(e)}")
            return {}
