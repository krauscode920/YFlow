import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import time
import sys
import traceback

# Updated imports to use the new multi-file structure
from yflow.core.model import Model
from yflow.layers.lstm import YSTM
from yflow.layers.dense import Dense
from yflow.losses.mse import MSELoss
from yflow.optimizers.adam import Adam

# More robust GPU import
try:
    from yflow.utils import is_gpu_available, get_device_info
except ImportError:
    def is_gpu_available():
        return False


    def get_device_info():
        return {
            'device_name': 'CPU',
            'memory_total': None,
            'memory_free': None
        }


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

    # Move model to GPU if available
    device_type = 'gpu' if is_gpu_available() else 'cpu'
    model.to(device_type)

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
        # Robust device detection
        device_type = 'gpu' if is_gpu_available() else 'cpu'
        print(f"\nDevice Information:")
        print("-" * 50)

        device_info = get_device_info()
        if device_type == 'gpu':
            print(f"Using GPU: {device_info['device_name']}")
            print(f"Available memory: {device_info['memory_free'] / 1e9:.2f} GB" if device_info[
                                                                                        'memory_free'] is not None else "GPU memory info unavailable")
        else:
            print("Using CPU: GPU not available")
        print("-" * 50 + "\n")

        # Prepare data
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler = prepare_data(
            seq_length=60,
            train_split=0.8
        )

        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        print(f"Target shape: {y_train.shape}, {y_test.shape}")

        # Create and train model
        model = create_model()

        with Timer("Model training"):
            history = model.train(
                X_train,
                y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                early_stopping=True,
                patience=5,
                verbose=1
            )

        # Make predictions and evaluate
        with Timer("Prediction"):
            predictions = model.predict(X_test)

        predictions, y_test_orig = evaluate_predictions(predictions, y_test, target_scaler)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nDetailed Error Traceback:")
        traceback.print_exc()
        sys.exit(1)