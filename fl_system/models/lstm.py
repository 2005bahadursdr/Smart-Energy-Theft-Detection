"""
PyTorch LSTM Architecture for Smart Grid Energy Theft Detection.
Designed for 24-hour and 48-hour sequence classification with stacked LSTM layers,
temporal aggregation, and regularization to address class imbalance.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class EnergyTheftLSTM(nn.Module):
    """
    Recurrent Neural Network leveraging Long Short-Term Memory (LSTM) cells
    to capture temporal consumption trajectories, anomalies, and theft signatures.
    """

    def __init__(
        self,
        input_size: int = 11,          # Matches multivariate feature dimensionality
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.20,
        fc_dims: Optional[List[int]] = None,
        output_size: int = 1
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        fc_dims = fc_dims or [32]

        # Stacked LSTM Layers
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden_size * self.num_directions

        # Layer normalization for temporal representation stability
        self.layer_norm = nn.LayerNorm(lstm_out_dim)

        # Fully Connected Classification Head
        layers = []
        in_dim = lstm_out_dim
        for dim in fc_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = dim

        layers.append(nn.Linear(in_dim, output_size))
        self.classifier = nn.Sequential(*layers)

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
            Raw binary classification logits of shape (batch_size,).
        """
        if x.dim() == 2:
            # Add feature dimension if univariate (batch_size, sequence_length, 1)
            x = x.unsqueeze(-1)

        # lstm_out: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Context vector: Average pooling over temporal sequence steps + last step
        avg_pool = torch.mean(lstm_out, dim=1)
        last_step = lstm_out[:, -1, :]
        context = self.layer_norm(0.5 * (avg_pool + last_step))

        # Class logits
        logits = self.classifier(context).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate class probabilities via sigmoid activation."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
