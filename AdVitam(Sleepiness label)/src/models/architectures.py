"""
LSTM Model Architectures for KSS Prediction

This module contains different LSTM model architectures for predicting
Karolinska Sleepiness Scale (KSS) scores from physiological features.
"""

import torch
import torch.nn as nn


class LSTMKSSModel(nn.Module):
    """
    Basic LSTM model for KSS prediction using PyTorch
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMKSSModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size)

        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Batch normalization
        last_output = self.batch_norm(last_output)

        # Dense layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)

        return out.squeeze()


class ImprovedLSTMKSSModel(nn.Module):
    """
    Improved LSTM model for KSS prediction with better architecture
    """

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.5):
        super(ImprovedLSTMKSSModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM for better feature extraction
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,  # Bidirectional for better context
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Batch normalization
        self.batch_norm1 = nn.BatchNorm1d(hidden_size * 2)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)

        # Dense layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)

        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Self-attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling over sequence dimension
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_size * 2)

        # Batch normalization
        pooled = self.batch_norm1(pooled)

        # Dense layers with residual connections
        out1 = self.leaky_relu(self.fc1(pooled))
        out1 = self.batch_norm2(out1)
        out1 = self.dropout1(out1)

        out2 = self.leaky_relu(self.fc2(out1))
        out2 = self.batch_norm3(out2)
        out2 = self.dropout2(out2)

        out3 = self.leaky_relu(self.fc3(out2))
        out3 = self.dropout3(out3)

        # Final output
        out = self.fc4(out3)

        return out.squeeze()


def create_lstm_model(
    input_shape, hidden_size=64, num_layers=2, dropout=0.3, learning_rate=0.001
):
    """
    Create a basic LSTM model for KSS prediction using PyTorch.

    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (n_windows, n_features)
    hidden_size : int
        Number of LSTM hidden units
    num_layers : int
        Number of LSTM layers
    dropout : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer

    Returns:
    --------
    model : LSTMKSSModel
        PyTorch LSTM model
    """
    n_windows, n_features = input_shape

    model = LSTMKSSModel(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    return model


def create_improved_lstm_model(
    input_shape, hidden_size=128, num_layers=3, dropout=0.5, learning_rate=0.0001
):
    """
    Create an improved LSTM model for KSS prediction with better architecture.

    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (n_windows, n_features)
    hidden_size : int
        Number of LSTM hidden units (increased)
    num_layers : int
        Number of LSTM layers (increased)
    dropout : float
        Dropout rate for regularization (increased)
    learning_rate : float
        Learning rate for optimizer (decreased)

    Returns:
    --------
    model : ImprovedLSTMKSSModel
        Improved PyTorch LSTM model
    """
    n_windows, n_features = input_shape

    model = ImprovedLSTMKSSModel(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    return model
