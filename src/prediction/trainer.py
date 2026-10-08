from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from .lstm_model import TrafficLSTMModel


# ============================================================
# TRAINING RESULT
# ============================================================

@dataclass(frozen=True)
class TrainingResult:
    train_losses: Tuple[float, ...]
    validation_losses: Tuple[float, ...]


# ============================================================
# LSTM TRAINER
# ============================================================

class TrafficLSTMTrainer:
    """
    Training pipeline for the traffic prediction LSTM.

    Input:
        X -> (samples, sequence_length, features)

    Target:
        y -> (samples, features)
    """

    def __init__(
        self,
        model: TrafficLSTMModel,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ) -> None:

        self.model = model

        self.device = torch.device(
            device
        )

        self.model.to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.MSELoss()

    # ========================================================
    # NUMPY -> TORCH
    # ========================================================

    def _to_tensor(
        self,
        values: np.ndarray,
    ) -> torch.Tensor:

        return torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

    # ========================================================
    # TRAIN ONE EPOCH
    # ========================================================

    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:

        self.model.train()

        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y)

        self.optimizer.zero_grad()

        predictions = self.model(
            X_tensor
        )

        loss = self.loss_function(
            predictions,
            y_tensor,
        )

        loss.backward()

        self.optimizer.step()

        return float(
            loss.detach().cpu().item()
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:

        self.model.eval()

        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y)

        with torch.no_grad():

            predictions = self.model(
                X_tensor
            )

            loss = self.loss_function(
                predictions,
                y_tensor,
            )

        return float(
            loss.detach().cpu().item()
        )

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        epochs: int = 20,
    ) -> TrainingResult:

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        train_losses = []
        validation_losses = []

        for _ in range(epochs):

            train_loss = self.train_epoch(
                X_train,
                y_train,
            )

            validation_loss = self.validate(
                X_validation,
                y_validation,
            )

            train_losses.append(
                train_loss
            )

            validation_losses.append(
                validation_loss
            )

        return TrainingResult(
            train_losses=tuple(
                train_losses
            ),
            validation_losses=tuple(
                validation_losses
            ),
        )