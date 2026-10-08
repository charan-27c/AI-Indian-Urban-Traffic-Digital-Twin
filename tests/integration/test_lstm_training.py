import numpy as np

from src.prediction.lstm_model import (
    TrafficLSTMModel,
)

from src.prediction.trainer import (
    TrafficLSTMTrainer,
)


def test_lstm_training_pipeline():

    # --------------------------------------------------------
    # Synthetic traffic sequences
    # --------------------------------------------------------

    X = np.random.rand(
        20,
        5,
        9,
    ).astype(
        np.float32
    )

    y = np.random.rand(
        20,
        9,
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    X_train = X[:16]
    y_train = y[:16]

    X_validation = X[16:]
    y_validation = y[16:]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TrafficLSTMModel()

    trainer = TrafficLSTMTrainer(
        model=model,
        learning_rate=0.001,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    result = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        epochs=3,
    )

    # --------------------------------------------------------
    # Verify training history
    # --------------------------------------------------------

    assert len(
        result.train_losses
    ) == 3

    assert len(
        result.validation_losses
    ) == 3

    # --------------------------------------------------------
    # Loss must be finite
    # --------------------------------------------------------

    assert all(
        np.isfinite(loss)
        for loss in result.train_losses
    )

    assert all(
        np.isfinite(loss)
        for loss in result.validation_losses
    )