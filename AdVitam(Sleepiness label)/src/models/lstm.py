"""
LSTM Pipeline for KSS Prediction

This module contains the complete LSTM pipeline for training, evaluating,
and making predictions with KSS models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional
import logging

# Import our modules using absolute imports
from src.models.architectures import create_lstm_model, create_improved_lstm_model

# Setup logger
logger = logging.getLogger(__name__)


class LSTMPipeline:
    """
    Complete LSTM pipeline for KSS prediction
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LSTM pipeline with configuration.

        Parameters:
        -----------
        config : dict
            Configuration dictionary containing model and training parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.train_scaler = None

        logger.info(f"Using device: {self.device}")
        logger.info(f"Configuration: {config}")

    def _prepare_data(
        self, X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for training by converting to tensors and moving to device.
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        participant_tensor = torch.LongTensor(participant_ids).to(self.device)

        return X_tensor, y_tensor, participant_tensor

    def _create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch data loaders for training, validation, and test sets.
        """
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(y_train).to(self.device),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.FloatTensor(y_val).to(self.device),
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test).to(self.device),
            torch.FloatTensor(y_test).to(self.device),
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _split_data_by_participant(
        self, X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data by participant to avoid data leakage.
        """
        # Use GroupShuffleSplit to ensure participants don't appear in multiple splits
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        # First split: separate test set
        train_val_idx, test_idx = next(gss.split(X, y, groups=participant_ids))

        # Second split: separate validation from training
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=42
        )  # 0.25 of remaining = 0.2 of total
        train_idx, val_idx = next(
            gss_val.split(
                X[train_val_idx],
                y[train_val_idx],
                groups=participant_ids[train_val_idx],
            )
        )

        # Map back to original indices
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]

        return train_idx, val_idx, test_idx

    def get_test_indices(
        self, X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray
    ) -> np.ndarray:
        """
        Get test set indices for proper evaluation.

        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        participant_ids : np.ndarray
            Participant IDs

        Returns:
        --------
        test_idx : np.ndarray
            Test set indices
        """
        _, _, test_idx = self._split_data_by_participant(X, y, participant_ids)
        return test_idx

    def _scale_features(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler fitted on training data.
        """
        # Fit scaler on training data only
        self.train_scaler = StandardScaler()
        X_train_scaled = self.train_scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)

        # Transform validation and test data
        X_val_scaled = self.train_scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)
        X_test_scaled = self.train_scaler.transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)

        return X_train_scaled, X_val_scaled, X_test_scaled

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        """
        Train for one epoch.
        """
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(
        self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module
    ) -> float:
        """
        Validate for one epoch.
        """
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _evaluate_model(
        self, model: nn.Module, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate model and return predictions and true values.
        """
        model.eval()
        predictions = []
        true_values = []

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                outputs = model(batch_X)
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(batch_y.cpu().numpy())

        return np.array(predictions), np.array(true_values)

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def train(
        self, X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray, logger=None
    ) -> Tuple[nn.Module, Dict[str, list], Dict[str, float], Dict[str, np.ndarray]]:
        """
        Train the LSTM model.

        Parameters:
        -----------
        X : np.ndarray
            Input features of shape (n_samples, n_windows, n_features)
        y : np.ndarray
            Target KSS scores of shape (n_samples,)
        participant_ids : np.ndarray
            Participant IDs for proper splitting
        logger : logging.Logger, optional
            Logger instance for detailed progress reporting

        Returns:
        --------
        model : nn.Module
            Trained model
        history : dict
            Training history
        metrics : dict
            Final evaluation metrics
        data_splits : dict
            Dictionary containing train/val/test splits of y values
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        logger.info("Starting model training...")
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Number of participants: {len(np.unique(participant_ids))}")

        # Split data by participant
        train_idx, val_idx, test_idx = self._split_data_by_participant(
            X, y, participant_ids
        )

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Store data splits for plotting
        data_splits = {"y_train": y_train, "y_val": y_val, "y_test": y_test}

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self._scale_features(
            X_train, X_val, X_test
        )

        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            X_test_scaled,
            y_test,
            batch_size=int(self.config.get("batch_size", 32)),
        )

        # Create model
        if self.config.get("use_improved_model", False):
            self.model = create_improved_lstm_model(
                input_shape=(X.shape[1], X.shape[2]),
                hidden_size=int(self.config.get("hidden_size", 128)),
                num_layers=int(self.config.get("num_layers", 3)),
                dropout=float(self.config.get("dropout", 0.5)),
            )
        else:
            self.model = create_lstm_model(
                input_shape=(X.shape[1], X.shape[2]),
                hidden_size=int(self.config.get("hidden_size", 64)),
                num_layers=int(self.config.get("num_layers", 2)),
                dropout=float(self.config.get("dropout", 0.3)),
            )

        self.model.to(self.device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        initial_lr = float(self.config.get("learning_rate", 0.001))
        weight_decay = float(self.config.get("weight_decay", 1e-5))

        logger.info(f"DEBUG: Initial learning rate from config: {initial_lr}")
        logger.info(f"DEBUG: Weight decay from config: {weight_decay}")

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        scheduler_t0 = int(self.config.get("scheduler_t0", 10))
        scheduler_t_mult = int(self.config.get("scheduler_t_mult", 2))

        logger.info(f"DEBUG: Scheduler T_0: {scheduler_t0}")
        logger.info(f"DEBUG: Scheduler T_mult: {scheduler_t_mult}")

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_t0,
            T_mult=scheduler_t_mult,
        )

        # Training loop
        num_epochs = int(self.config.get("num_epochs", 100))
        patience = int(self.config.get("patience", 20))
        best_val_loss = float("inf")
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "train_r2": [],
            "val_r2": [],
            "lr": [],
        }

        logger.info("Starting training loop...")
        for epoch in range(num_epochs):
            # Train
            train_loss = self._train_epoch(
                self.model, train_loader, criterion, optimizer
            )

            # Validate
            val_loss = self._validate_epoch(self.model, val_loader, criterion)

            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # Calculate metrics for this epoch
            with torch.no_grad():
                # Training metrics
                y_pred_train, y_true_train = self._evaluate_model(
                    self.model, train_loader
                )
                train_metrics = self._calculate_metrics(y_true_train, y_pred_train)

                # Validation metrics
                y_pred_val, y_true_val = self._evaluate_model(self.model, val_loader)
                val_metrics = self._calculate_metrics(y_true_val, y_pred_val)

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_mae"].append(train_metrics["mae"])
            history["val_mae"].append(val_metrics["mae"])
            history["train_r2"].append(train_metrics["r2"])
            history["val_r2"].append(val_metrics["r2"])
            history["lr"].append(current_lr)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1

            # Print progress every 10 epochs (only loss for cleaner output)
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"LR: {current_lr:.6f}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        self.model.load_state_dict(torch.load("best_model.pth"))

        # Evaluate on validation set
        y_pred_val, y_true_val = self._evaluate_model(self.model, val_loader)
        val_metrics = self._calculate_metrics(y_true_val, y_pred_val)

        # Evaluate on test set
        y_pred_test, y_true_test = self._evaluate_model(self.model, test_loader)
        test_metrics = self._calculate_metrics(y_true_test, y_pred_test)

        # Calculate training metrics (on training set)
        y_pred_train, y_true_train = self._evaluate_model(self.model, train_loader)
        train_metrics = self._calculate_metrics(y_true_train, y_pred_train)

        # Combine all metrics
        all_metrics = {
            # Training metrics
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            # Validation metrics
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            # Test metrics (for backward compatibility)
            "mse": test_metrics["mse"],
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "r2": test_metrics["r2"],
            # Test metrics (explicit)
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        }

        logger.info("Training completed!")

        return self.model, history, all_metrics, data_splits

    def evaluate(
        self,
        model_path: str,
        X: np.ndarray,
        y: np.ndarray,
        participant_ids: np.ndarray,
        logger=None,
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.

        Parameters:
        -----------
        model_path : str
            Path to the trained model
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        participant_ids : np.ndarray
            Participant IDs
        logger : logging.Logger, optional
            Logger instance for detailed result reporting

        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        logger.info("Starting model evaluation...")

        # Load model
        if self.model is None:
            self.model = create_lstm_model(
                input_shape=(X.shape[1], X.shape[2]),
                hidden_size=int(self.config.get("hidden_size", 64)),
                num_layers=int(self.config.get("num_layers", 2)),
                dropout=float(self.config.get("dropout", 0.3)),
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)

        # Split data
        train_idx, val_idx, test_idx = self._split_data_by_participant(
            X, y, participant_ids
        )

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self._scale_features(
            X_train, X_val, X_test
        )

        # Create data loaders
        _, val_loader, test_loader = self._create_data_loaders(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            X_test_scaled,
            y_test,
            batch_size=int(self.config.get("batch_size", 32)),
        )

        # Evaluate on test set
        y_pred_test, y_true_test = self._evaluate_model(self.model, test_loader)
        metrics = self._calculate_metrics(y_true_test, y_pred_test)

        # Print detailed test results
        logger.info("TEST RESULTS")
        logger.info(f"Test samples: {len(y_true_test)}")
        logger.info(
            f"True KSS range: {y_true_test.min():.2f} - {y_true_test.max():.2f}"
        )
        logger.info(
            f"Predicted KSS range: {y_pred_test.min():.2f} - {y_pred_test.max():.2f}"
        )
        logger.info(f"MSE: {metrics['mse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")

        return metrics

    def predict(self, model_path: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model.

        Parameters:
        -----------
        model_path : str
            Path to the trained model
        X : np.ndarray
            Input features

        Returns:
        --------
        predictions : np.ndarray
            Model predictions
        """
        # Load model if not already loaded
        if self.model is None:
            self.model = create_lstm_model(
                input_shape=(X.shape[1], X.shape[2]),
                hidden_size=int(self.config.get("hidden_size", 64)),
                num_layers=int(self.config.get("num_layers", 2)),
                dropout=float(self.config.get("dropout", 0.3)),
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)

        # Scale features if scaler is available
        if self.train_scaler is not None:
            X_scaled = self.train_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(
                X.shape
            )
        else:
            X_scaled = X

        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(X_scaled).to(self.device))
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Make predictions
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for (batch_X,) in data_loader:
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")

    def save_results(
        self, metrics: Dict[str, float], history: Dict[str, list], output_dir: Path
    ):
        """Save training results and plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save training history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save scaler
        if self.train_scaler is not None:
            with open(output_dir / "scaler.pkl", "wb") as f:
                pickle.dump(self.train_scaler, f)

        # Note: Training plots are now generated by KSSPlotter in main.py
        # to avoid duplication with training_history.png

        logger.info(f"Results saved to {output_dir}")

    def save_evaluation_results(self, metrics: Dict[str, float], output_dir: Path):
        """Save evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Evaluation results saved to {output_dir}")
