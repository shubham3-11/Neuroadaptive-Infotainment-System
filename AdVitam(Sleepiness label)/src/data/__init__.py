"""
Data Module for KSS Prediction Pipeline

This module provides data loading and processing for KSS prediction.

Usage:
    from src.data import AdVitamDataLoader
"""

# Data loading utilities
from .advitam_loader import AdVitamDataLoader

__all__ = ["AdVitamDataLoader"]

# Version information
__version__ = "2.0.0"
__author__ = "KSS Prediction Team"
__description__ = "Data processing for KSS prediction"
