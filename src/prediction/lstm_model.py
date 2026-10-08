from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


# ============================================================
# LSTM CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class LSTMConfig:
    """
    Configuration for the traffic prediction LSTM.

    Input:
        (batch, sequence_length, feature_count)

    Output:
        (batch, feature_count)
    """

    sequence_length: int = 5
    feature_count: int = 9

    hidden_size: int = 64
    num_layers: int = 2

    dropout: float = 0.2

    learning_rate: float = 0.001


# ============================================================
# TRAFFIC LSTM MODEL
# ============================================================

class TrafficLSTMModel:
    """
    PyTorch-backed LSTM wrapper for traffic-state prediction.

    The wrapper keeps the project-level prediction API clean while
    exposing the common PyTorch module operations required by the
    training pipeline.

    Input shape:

        (batch, sequence_length, feature_count)

    Output shape:

        (batch, feature_count)
    """

    def __init__(
        self,
        config: Optional[LSTMConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else LSTMConfig()
        )

        self._validate_config()

        self._model = None

        # Build lazily so importing the prediction package does
        # not require PyTorch immediately.
        self.build()

    # ========================================================
    # CONFIG VALIDATION
    # ========================================================

    def _validate_config(self) -> None:

        if self.config.sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than zero."
            )

        if self.config.feature_count <= 0:
            raise ValueError(
                "feature_count must be greater than zero."
            )

        if self.config.hidden_size <= 0:
            raise ValueError(
                "hidden_size must be greater than zero."
            )

        if self.config.num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than zero."
            )

        if not 0.0 <= self.config.dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        if self.config.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    def build(self):
        """
        Build the underlying PyTorch model.
        """

        import torch.nn as nn

        config = self.config

        class _TrafficLSTM(nn.Module):

            def __init__(self):
                super().__init__()

                self.lstm = nn.LSTM(
                    input_size=config.feature_count,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    batch_first=True,
                    dropout=(
                        config.dropout
                        if config.num_layers > 1
                        else 0.0
                    ),
                )

                self.output = nn.Sequential(

                    nn.Linear(
                        config.hidden_size,
                        config.hidden_size,
                    ),

                    nn.ReLU(),

                    nn.Linear(
                        config.hidden_size,
                        config.feature_count,
                    ),
                )

            def forward(self, x):

                output, _ = self.lstm(x)

                # Last timestep
                last_output = output[:, -1, :]

                return self.output(
                    last_output
                )

        self._model = _TrafficLSTM()

        return self._model

    # ========================================================
    # MODEL PROPERTY
    # ========================================================

    @property
    def model(self):
        """
        Return underlying torch.nn.Module.
        """

        if self._model is None:
            self.build()

        return self._model

    # ========================================================
    # PYTORCH COMPATIBILITY
    # ========================================================

    def to(self, device):
        """
        Move model to CPU/GPU device.
        """

        self.model.to(device)

        return self

    def train(self, mode: bool = True):
        """
        Set training/evaluation mode.
        """

        self.model.train(mode)

        return self

    def eval(self):
        """
        Set evaluation mode.
        """

        self.model.eval()

        return self

    def parameters(self):
        """
        Return trainable model parameters.

        Required by optimizers such as Adam.
        """

        return self.model.parameters()

    def named_parameters(self):
        """
        Return named trainable parameters.
        """

        return self.model.named_parameters()

    def state_dict(self):
        """
        Return PyTorch model state dictionary.
        """

        return self.model.state_dict()

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
    ):
        """
        Load PyTorch model weights.
        """

        return self.model.load_state_dict(
            state_dict,
            strict=strict,
        )

    def __call__(self, x):
        """
        Forward call.

        Allows:

            model(x)

        exactly like a normal PyTorch model.
        """

        return self.model(x)

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def validate_input(
        self,
        X: np.ndarray,
    ) -> None:

        X = np.asarray(X)

        if X.ndim != 3:

            raise ValueError(
                "LSTM input must have 3 dimensions: "
                "(batch, sequence_length, feature_count)."
            )

        if X.shape[1] != (
            self.config.sequence_length
        ):

            raise ValueError(
                "Invalid sequence length. "
                f"Expected "
                f"{self.config.sequence_length}, "
                f"received {X.shape[1]}."
            )

        if X.shape[2] != (
            self.config.feature_count
        ):

            raise ValueError(
                "Invalid feature count. "
                f"Expected "
                f"{self.config.feature_count}, "
                f"received {X.shape[2]}."
            )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the next traffic state.

        Input:

            (batch, sequence_length, feature_count)

        Output:

            (batch, feature_count)
        """

        import torch

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        self.validate_input(X)

        self.model.eval()

        tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        with torch.no_grad():

            prediction = self.model(
                tensor
            )

        return prediction.cpu().numpy()

    # ========================================================
    # SINGLE SEQUENCE PREDICTION
    # ========================================================

    def predict_sequence(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict one traffic sequence.

        Input:

            (sequence_length, feature_count)

        Output:

            (feature_count,)
        """

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 2:

            raise ValueError(
                "Single sequence must have shape "
                "(sequence_length, feature_count)."
            )

        if X.shape[0] != (
            self.config.sequence_length
        ):

            raise ValueError(
                "Invalid sequence length. "
                f"Expected "
                f"{self.config.sequence_length}, "
                f"received {X.shape[0]}."
            )

        if X.shape[1] != (
            self.config.feature_count
        ):

            raise ValueError(
                "Invalid feature count. "
                f"Expected "
                f"{self.config.feature_count}, "
                f"received {X.shape[1]}."
            )

        X_batch = np.expand_dims(
            X,
            axis=0,
        )

        prediction = self.predict(
            X_batch
        )

        return prediction[0]

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path: str,
    ) -> None:
        """
        Save model configuration and weights.
        """

        import torch

        torch.save(
            {
                "config": self.config,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "TrafficLSTMModel":
        """
        Load a previously saved LSTM model.
        """

        import torch

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        config = checkpoint["config"]

        instance = cls(
            config=config
        )

        instance.model.load_state_dict(
            checkpoint["state_dict"]
        )

        instance.model.eval()

        return instance