# GPU_Test.py

from yflow.core.context import DeviceContext
import numpy as np


# Create model using global device context
def create_model():
    from yflow.core.model import Model
    from yflow.layers.dense import Dense

    model = Model()
    model.add(Dense(128, input_size=64))
    model.add(Dense(10))

    # Compile the model - THIS WAS MISSING
    from yflow.losses.mse import MSELoss
    from yflow.optimizers.adam import Adam
    model.compile(loss=MSELoss(), optimizer=Adam())

    return model


# Train on any device
def train_model(model, X, y, device_type='cpu'):
    with DeviceContext.device(device_type) as device:
        # Data automatically moves to correct device
        X = device.to_device(X)
        y = device.to_device(y)

        # Model operations use the device from context
        model.train(X, y, epochs=10, batch_size=32)

        # Results come back as NumPy arrays
        return device.to_cpu(model.predict(X))


# This code works identically on your MacBook and on Colab with GPU
if __name__ == "__main__":
    # Example data
    X = np.random.randn(1000, 64)
    y = np.random.randint(0, 10, size=(1000, 1))

    # Create model
    model = create_model()

    # Try to use GPU if available, otherwise CPU
    device_type = 'gpu' if DeviceContext.get_device().is_gpu_available() else 'cpu'
    print(f"Using device: {device_type}")

    # Train and predict
    predictions = train_model(model, X, y, device_type)
    print(f"Predictions shape: {predictions.shape}")