"""
PyTorch CNN-LSTM Hybrid Architecture for Smart Grid Energy Theft Detection.
Fuses 1D temporal convolutions (for localized pattern and transient drop extraction)
with LSTM recurrent units (for long-term temporal trajectory modeling).
"""

from typing import List, Optional
import torch
import torch.nn as nn


class EnergyTheftCNNLSTM(nn.Module):
    """
    Hybrid Convolutional-Recurrent Neural Network.
    1D Convolutions extract micro-features (sharp consumption dips, bypass flatlines, peak shaving)
    while the LSTM layers track macro-scale diurnal and weekly dependencies.
    """

    def __init__(
        self,
        input_size: int = 11,
        conv_filters: Optional[List[int]] = None,
        kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        dense_units: Optional[List[int]] = None,
        dropout: float = 0.25,
        output_size: int = 1
    ):
        super().__init__()
        self.input_size = input_size
        conv_filters = conv_filters or [32, 64]
        dense_units = dense_units or [32]

        # 1. 1D Convolutional Feature Extractor
        conv_modules = []
        in_channels = input_size
        for filters in conv_filters:
            conv_modules.append(nn.Conv1d(in_channels, filters, kernel_size=kernel_size, padding=kernel_size // 2))
            conv_modules.append(nn.BatchNorm1d(filters))
            conv_modules.append(nn.ReLU())
            conv_modules.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
            in_channels = filters

        self.conv_net = nn.Sequential(*conv_modules)

        # 2. Recurrent LSTM Unit
        self.lstm = nn.LSTM(
            input_size=conv_filters[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.layer_norm = nn.LayerNorm(lstm_hidden_size)

        # 3. Dense Classification Head
        dense_layers = []
        in_dim = lstm_hidden_size
        for units in dense_units:
            dense_layers.append(nn.Linear(in_dim, units))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout))
            in_dim = units

        dense_layers.append(nn.Linear(in_dim, output_size))
        self.classifier = nn.Sequential(*dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            Binary classification logits of shape (batch_size,).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # PyTorch Conv1d expects (batch_size, channels, sequence_length)
        x_perm = x.permute(0, 2, 1)

        # Extract localized temporal features
        conv_out = self.conv_net(x_perm)

        # Permute back to (batch_size, reduced_sequence_length, channels) for LSTM
        lstm_in = conv_out.permute(0, 2, 1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(lstm_in)

        # Temporal pooling (average of last two steps or mean pool)
        if lstm_out.size(1) >= 2:
            pool = torch.mean(lstm_out[:, -2:, :], dim=1)
        else:
            pool = lstm_out[:, -1, :]

        context = self.layer_norm(pool)

        # Classification logits
        logits = self.classifier(context).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate class probabilities via sigmoid activation."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
