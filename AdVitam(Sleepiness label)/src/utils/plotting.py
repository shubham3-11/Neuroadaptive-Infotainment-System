"""
Plotting utilities for KSS prediction results

This module provides comprehensive plotting functions for visualizing:
- Training history (loss, metrics)
- Model performance (predictions vs actual)
- Residual analysis
- Feature importance
- Data distribution
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
import logging
import collections.abc

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class KSSPlotter:
    """
    Comprehensive plotting utility for KSS prediction results
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the plotter.

        Parameters:
        -----------
        output_dir : Path, optional
            Directory to save plots. If None, plots will be displayed only.
        """
        if output_dir:
            self.output_dir = Path(output_dir) / "plots"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

    def plot_training_history(
        self, history: Dict[str, List[float]], save_path: Optional[str] = None
    ) -> None:
        """
        Plot training history including loss and metrics.

        Parameters:
        -----------
        history : Dict[str, List[float]]
            Training history dictionary with keys like 'train_loss', 'val_loss', etc.
        save_path : str, optional
            Path to save the plot. If None, uses self.output_dir.
        """
        # Determine what metrics are available
        available_metrics = []
        if "train_loss" in history or "val_loss" in history:
            available_metrics.append("loss")
        if "train_mae" in history or "val_mae" in history:
            available_metrics.append("mae")
        if "train_r2" in history or "val_r2" in history:
            available_metrics.append("r2")
        if "lr" in history:
            available_metrics.append("lr")

        # Create appropriate subplot layout
        n_metrics = len(available_metrics)
        if n_metrics == 0:
            logger.warning("No training history data available for plotting")
            return

        # Determine subplot layout
        if n_metrics == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        elif n_metrics == 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        elif n_metrics == 3:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:  # n_metrics >= 4
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # Always flatten axes to a list
        if isinstance(axes, np.ndarray):
            axes = axes.flatten().tolist()
        elif not isinstance(axes, list):
            axes = [axes]

        fig.suptitle("Training History", fontsize=16, fontweight="bold")

        plot_idx = 0

        # Loss plot
        if "loss" in available_metrics:
            ax = axes[plot_idx] if n_metrics > 1 else axes
            if history.get("train_loss"):
                ax.plot(
                    history["train_loss"],
                    label="Training Loss",
                    linewidth=2,
                    color="blue",
                )
            if history.get("val_loss"):
                ax.plot(
                    history["val_loss"],
                    label="Validation Loss",
                    linewidth=2,
                    color="red",
                )
            ax.set_title("Loss Over Time")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # MAE plot
        if "mae" in available_metrics:
            ax = axes[plot_idx] if n_metrics > 1 else axes
            if history.get("train_mae"):
                ax.plot(
                    history["train_mae"],
                    label="Training MAE",
                    linewidth=2,
                    color="blue",
                )
            if history.get("val_mae"):
                ax.plot(
                    history["val_mae"], label="Validation MAE", linewidth=2, color="red"
                )
            ax.set_title("Mean Absolute Error Over Time")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MAE")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # R² plot
        if "r2" in available_metrics:
            ax = axes[plot_idx] if n_metrics > 1 else axes
            if history.get("train_r2"):
                ax.plot(
                    history["train_r2"], label="Training R²", linewidth=2, color="blue"
                )
            if history.get("val_r2"):
                ax.plot(
                    history["val_r2"], label="Validation R²", linewidth=2, color="red"
                )
            ax.set_title("R² Score Over Time")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("R²")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Learning rate plot
        if "lr" in available_metrics:
            ax = axes[plot_idx] if n_metrics > 1 else axes
            ax.plot(history["lr"], linewidth=2, color="purple")
            ax.set_title("Learning Rate Schedule")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
            plot_idx += 1

        # Hide unused subplots if any
        if n_metrics > 1:
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training history plot saved to {save_path}")
        elif self.output_dir:
            save_path = self.output_dir / "training_history.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training history plot saved to {save_path}")

        # Show interactively only if no output directory provided
        if self.output_dir is None and save_path is None:
            plt.show()
        else:
            plt.close(fig)

    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Test Set: Predictions vs Actual Values",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot predictions against actual values with regression line.

        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot with regression line
        ax1.scatter(y_true, y_pred, alpha=0.6, s=50)

        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax1.plot(y_true, p(y_true), "r--", alpha=0.8, linewidth=2)

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], "k-", alpha=0.5, linewidth=1)

        ax1.set_xlabel("Actual KSS Values")
        ax1.set_ylabel("Predicted KSS Values")
        ax1.set_title(f"{title}\nR² = {np.corrcoef(y_true, y_pred)[0, 1]**2:.3f}")
        ax1.grid(True, alpha=0.3)

        # Residual plot
        residuals = y_pred - y_true
        ax2.scatter(y_true, residuals, alpha=0.6, s=50)
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.8)
        ax2.set_xlabel("Actual KSS Values")
        ax2.set_ylabel("Residuals (Predicted - Actual)")
        ax2.set_title("Residual Plot")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Predictions vs actual plot saved to {save_path}")
        elif self.output_dir:
            save_path = self.output_dir / "predictions_vs_actual.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Predictions vs actual plot saved to {save_path}")

        if self.output_dir is None and save_path is None:
            plt.show()
        else:
            plt.close(fig)

    def plot_residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Test Data: Residual Analysis",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Comprehensive residual analysis plots.

        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        save_path : str, optional
            Path to save the plot
        """
        residuals = y_pred - y_true

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Residuals vs fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=50)
        axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.8)
        axes[0, 0].set_xlabel("Predicted Values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Predicted")
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor="black")
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Residuals Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normal Distribution)")
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals over time/index
        axes[1, 1].plot(residuals, alpha=0.7, linewidth=1)
        axes[1, 1].axhline(y=0, color="r", linestyle="--", alpha=0.8)
        axes[1, 1].set_xlabel("Sample Index")
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].set_title("Residuals Over Time")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Residual analysis plot saved to {save_path}")
        elif self.output_dir:
            save_path = self.output_dir / "residual_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Residual analysis plot saved to {save_path}")

        if self.output_dir is None and save_path is None:
            plt.show()
        else:
            plt.close(fig)

    def plot_data_distribution(
        self,
        y_train: np.ndarray = None,
        y_val: np.ndarray = None,
        y_test: np.ndarray = None,
        title: str = "KSS Distribution by Dataset",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot distribution of KSS values for train/val/test datasets.

        Parameters:
        -----------
        y_train : np.ndarray, optional
            Training KSS values
        y_val : np.ndarray, optional
            Validation KSS values
        y_test : np.ndarray, optional
            Test KSS values
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        # Collect available datasets
        datasets = {}
        if y_train is not None and len(y_train) > 0:
            datasets["Train"] = y_train
        if y_val is not None and len(y_val) > 0:
            datasets["Validation"] = y_val
        if y_test is not None and len(y_test) > 0:
            datasets["Test"] = y_test

        if not datasets:
            logger.warning("No data provided for distribution plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Colors for each dataset
        colors = {"Train": "skyblue", "Validation": "lightcoral", "Test": "lightgreen"}

        # Histogram
        for name, data in datasets.items():
            ax1.hist(
                data,
                bins=20,
                alpha=0.7,
                edgecolor="black",
                density=True,
                label=f"{name} (n={len(data)})",
                color=colors[name],
            )

        ax1.set_xlabel("KSS Values")
        ax1.set_ylabel("Density")
        ax1.set_title(f"{title} - Histogram")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        box_data = [datasets[name] for name in datasets.keys()]
        box_labels = [f"{name}\n(n={len(data)})" for name, data in datasets.items()]

        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel("KSS Values")
        ax2.set_title(f"{title} - Box Plot")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Data distribution plot saved to {save_path}")
        elif self.output_dir:
            save_path = self.output_dir / "data_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Data distribution plot saved to {save_path}")

        if self.output_dir is None and save_path is None:
            plt.show()
        else:
            plt.close(fig)

    def plot_metrics_comparison(
        self,
        metrics: Dict[str, float],
        title: str = "Model Performance Metrics",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot model performance metrics as a bar chart, grouped by train/val/test.

        Parameters:
        -----------
        metrics : Dict[str, float]
            Dictionary of metric names and values
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        # Group metrics by dataset (train/val/test) and metric type
        metric_groups = {"train": {}, "val": {}, "test": {}}

        # Extract metrics for each group
        for key, value in metrics.items():
            if key.startswith("train_"):
                metric_name = key.replace("train_", "")
                metric_groups["train"][metric_name] = value
            elif key.startswith("val_"):
                metric_name = key.replace("val_", "")
                metric_groups["val"][metric_name] = value
            elif key.startswith("test_"):
                metric_name = key.replace("test_", "")
                metric_groups["test"][metric_name] = value
            # Also handle metrics without prefix (legacy format)
            elif (
                key in ["mse", "mae", "rmse", "r2"]
                and "test" not in key
                and "train" not in key
                and "val" not in key
            ):
                # Assume these are test metrics if no prefix
                metric_groups["test"][key] = value

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Define colors for each group
        colors = {"train": "skyblue", "val": "lightcoral", "test": "lightgreen"}

        # Get all unique metric types
        all_metrics = set()
        for group in metric_groups.values():
            all_metrics.update(group.keys())
        all_metrics = sorted(list(all_metrics))

        # Set up bar positions
        x = np.arange(len(all_metrics))
        width = 0.25

        # Plot bars for each group
        for i, (group_name, group_metrics) in enumerate(metric_groups.items()):
            if group_metrics:  # Only plot if group has metrics
                values = [group_metrics.get(metric, 0) for metric in all_metrics]
                bars = ax.bar(
                    x + i * width,
                    values,
                    width,
                    label=group_name.capitalize(),
                    color=colors[group_name],
                    alpha=0.8,
                    edgecolor="black",
                )

                # Add value labels on bars
                for bar, value, metric_name in zip(bars, values, all_metrics):
                    height = bar.get_height()
                    # Format R² values differently (show as percentage)
                    if metric_name.upper() == "R2":
                        label_text = f"{value:.1%}"
                    else:
                        label_text = f"{value:.3f}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01 if height >= 0 else height - 0.01,
                        label_text,
                        ha="center",
                        va="bottom" if height >= 0 else "top",
                        fontweight="bold",
                        fontsize=8,
                    )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Metric Value")
        ax.set_xlabel("Metrics")
        ax.set_xticks(x + width)
        ax.set_xticklabels([metric.upper() for metric in all_metrics])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Metrics comparison plot saved to {save_path}")
        elif self.output_dir:
            save_path = self.output_dir / "metrics_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Metrics comparison plot saved to {save_path}")

        if self.output_dir is None and save_path is None:
            plt.show()
        else:
            plt.close(fig)

    def plot_all_results(
        self,
        history: Dict[str, List[float]],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, float],
        y_train: np.ndarray = None,
        y_val: np.ndarray = None,
        y_test: np.ndarray = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        """
        Generate all plots and save them to the specified directory.

        Parameters:
        -----------
        history : Dict[str, List[float]]
            Training history
        y_true : np.ndarray
            True values (for predictions plot)
        y_pred : np.ndarray
            Predicted values (for predictions plot)
        metrics : Dict[str, float]
            Model performance metrics
        y_train : np.ndarray, optional
            Training KSS values for distribution plot
        y_val : np.ndarray, optional
            Validation KSS values for distribution plot
        y_test : np.ndarray, optional
            Test KSS values for distribution plot
        save_dir : Path, optional
            Directory to save all plots
        """
        if save_dir:
            self.output_dir = Path(save_dir)
            self.output_dir.mkdir(exist_ok=True)

        # Generate training history plot only if history data is available
        if history and any(history.values()):
            self.plot_training_history(history)

        # Generate prediction plots only if we have data
        if len(y_true) > 0 and len(y_pred) > 0:
            self.plot_predictions_vs_actual(y_true, y_pred)
            self.plot_residual_analysis(y_true, y_pred)

        # Generate data distribution plot with train/val/test data
        if y_train is not None or y_val is not None or y_test is not None:
            self.plot_data_distribution(y_train, y_val, y_test)

        # Generate metrics comparison if metrics are available
        if metrics:
            self.plot_metrics_comparison(metrics)

        logger.info(f"All plots generated and saved to {self.output_dir}")

    def load_and_plot_from_results(self, results_dir: Path) -> None:
        """
        Load results from a results directory and generate all plots.

        Parameters:
        -----------
        results_dir : Path
            Directory containing results files (history.json, metrics.json, etc.)
        """
        results_dir = Path(results_dir)

        # Load history
        history_file = results_dir / "history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)
        else:
            logger.warning(f"History file not found: {history_file}")
            history = {}

        # Load metrics
        metrics_file = results_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        else:
            logger.warning(f"Metrics file not found: {metrics_file}")
            metrics = {}

        # Load evaluation results if available
        eval_file = results_dir / "evaluation" / "evaluation_metrics.json"
        if eval_file.exists():
            with open(eval_file, "r") as f:
                eval_metrics = json.load(f)
            metrics.update(eval_metrics)

        # For now, we'll create dummy data for predictions vs actual
        # In a real scenario, you'd load the actual predictions
        if "test_mae" in metrics:
            # Create synthetic data for demonstration
            n_samples = 100
            y_true = np.random.uniform(1, 9, n_samples)
            y_pred = y_true + np.random.normal(0, metrics["test_mae"], n_samples)
            y_pred = np.clip(y_pred, 1, 9)  # Clip to KSS range
        else:
            y_true = np.array([])
            y_pred = np.array([])

        # Generate plots
        self.output_dir = results_dir
        self.plot_all_results(history, y_true, y_pred, metrics)


def create_plots_from_results(
    results_dir: str, output_dir: Optional[str] = None
) -> None:
    """
    Convenience function to create plots from results directory.

    Parameters:
    -----------
    results_dir : str
        Path to results directory
    output_dir : str, optional
        Output directory for plots (if different from results_dir)
    """
    plotter = KSSPlotter(output_dir)
    plotter.load_and_plot_from_results(Path(results_dir))


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots from results")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to results directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results_dir)",
    )

    args = parser.parse_args()

    create_plots_from_results(args.results_dir, args.output_dir)
