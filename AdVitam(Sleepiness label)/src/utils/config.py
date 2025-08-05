"""
Configuration utilities for KSS prediction pipeline.
"""

from typing import Dict, Any
import yaml


class Config:
    """
    Simple configuration class for the KSS prediction pipeline.
    """

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration from dictionary.

        Parameters:
        -----------
        config_dict : dict, optional
            Configuration dictionary. If None, uses empty dict.
        """
        if config_dict is None:
            config_dict = {}

        # Set configuration sections
        self.data = config_dict.get("data", {})
        self.preprocessing = config_dict.get("preprocessing", {})
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})
        self.prediction = config_dict.get("prediction", {})

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        Create a Config instance from a YAML file.

        Parameters:
        -----------
        yaml_path : str
            Path to YAML configuration file

        Returns:
        --------
        config : Config
            Configuration instance
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
        --------
        config_dict : dict
            Configuration dictionary
        """
        return {
            "data": self.data,
            "preprocessing": self.preprocessing,
            "model": self.model,
            "training": self.training,
            "prediction": self.prediction,
        }

    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.

        Parameters:
        -----------
        yaml_path : str
            Path to save YAML configuration file
        """
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
