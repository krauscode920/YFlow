import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import time
import sys
import traceback

# Updated imports to use the new device context
from yflow.core.model import Model
from yflow.layers.lstm import YSTM
from yflow.layers.dense import Dense
from yflow.losses.mse import MSELoss
from yflow.optimizers.adam import Adam
from yflow.core.context import DeviceContext  # Updated import


# Sample data generation for testing
def generate_sample_stock_data(days=1000):
    """Generate synthetic stock data for testing"""
    np.random.seed(42)
    t = np.linspace(0, days, days)

    # Generate synthetic OHLCV data
    close = 100 + 10 * np.sin(t / 100) + t / 10 + np.random.normal(0, 5, days)
    high = close + np.random.uniform(0, 5, days)
    low = close - np.random.uniform(0, 5, days)
    open_price = (high + low) / 2 + np.random.normal(0, 2, days)
    volume = 1000000 + 500000 * np.sin(t / 50) + np.random.normal(0, 100000, days)

    # Combine into OHLCV array
    data = np.column_stack((open_price, high, low, close, volume))
    print(f"Generated {len(data)} days of synthetic stock data")
    return data


class Timer:
    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"{self.name} took {self.interval:.2f} seconds")


def create_sequences(data: np.ndarray,
                     seq_length: int = 60,
                     target_col: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        sequence = data[i:(i + seq_length)]
        target = data[i + seq_length, target_col]
        X.append(sequence)
        y.append(target)
    return np.array(X), np.array(y)


def prepare_data(seq_length: int = 60,
                 train_split: float = 0.8) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Prepare data for training"""
    with Timer("Data preparation"):
        # Use synthetic data instead of fetching
        data = generate_sample_stock_data()

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        scaled_data = feature_scaler.fit_transform(data)
        target_scaler.fit(data[:, [3]])  # Fit on Close prices

        X, y = create_sequences(scaled_data, seq_length)

        y = y.reshape(-1, 1)
        y = target_scaler.transform(y)

        train_size = int(len(X) * train_split)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler


def create_model():
    """Create and compile the stock prediction model with GPU support"""
    model = Model()

    # No need to explicitly move the model here - will be handled by context

    model.add(YSTM(
        input_size=5,  # OHLCV features
        hidden_size=64,
        layer_norm=True,
        dropout=0.2,
        return_sequences=False,
        batch_first=True
    ))

    model.add(Dense(
        output_size=1  # Predict next day's price
    ))

    model.compile(
        loss=MSELoss(),
        optimizer=Adam(learning_rate=0.001)
    )

    return model


def evaluate_predictions(predictions: np.ndarray, y_test: np.ndarray, target_scaler: MinMaxScaler):
    """Evaluate and print model performance metrics"""
    # Convert predictions back to original scale
    predictions = target_scaler.inverse_transform(predictions)
    y_test_orig = target_scaler.inverse_transform(y_test)

    # Calculate metrics
    mse = np.mean((predictions - y_test_orig) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_orig - predictions) / y_test_orig)) * 100

    print("\nModel Performance:")
    print("-" * 50)
    print(f"Test MSE: ${mse:.2f}")
    print(f"Test RMSE: ${rmse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    return predictions, y_test_orig


if __name__ == "__main__":
    try:
        # Check for GPU availability using DeviceContext
        device_context = DeviceContext.get_device()
        device_type = 'gpu' if device_context.is_gpu_available() else 'cpu'

        print(f"\nUsing device: {device_type}")

        # Prepare data
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler = prepare_data(
            seq_length=60,
            train_split=0.8
        )

        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        print(f"Target shape: {y_train.shape}, {y_test.shape}")

        # Training and prediction with device context
        with DeviceContext.device(device_type) as device:
            # Create model
            model = create_model()

            # Move data to device
            X_train_device = device.to_device(X_train)
            y_train_device = device.to_device(y_train)
            X_test_device = device.to_device(X_test)
            y_test_device = device.to_device(y_test)

            # Train model
            with Timer("Model training"):
                history = model.train(
                    X_train_device,
                    y_train_device,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test_device, y_test_device),
                    early_stopping=True,
                    patience=5,
                    verbose=1
                )

            # Make predictions
            with Timer("Prediction"):
                predictions_device = model.predict(X_test_device)
                predictions = device.to_cpu(predictions_device)

            # Clear GPU memory if applicable
            device.clear_memory()

        # Evaluate predictions (back on CPU)
        predictions, y_test_orig = evaluate_predictions(predictions, y_test, target_scaler)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nDetailed Error Traceback:")
        traceback.print_exc()
        sys.exit(1)