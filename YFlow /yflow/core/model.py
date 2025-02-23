import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import random
from .shape_handler import ShapeHandler
from .layer import Layer
from .device import Device

def _set_global_seed(seed=42):
    """Set the seed for all random number generators used in the library."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import cupy as cp
        cp.random.seed(seed)
    except ImportError:
        pass

class Model:
    """Model class with improved shape handling and device support"""

    def __init__(self, seed: Optional[int] = None):
        self.layers: List[Layer] = []
        self.loss = None
        self.optimizer = None
        self.shape_handler = ShapeHandler()
        self.seed = 42 if seed is None else seed
        self.device = Device('cpu')  # Default to CPU
        _set_global_seed(self.seed)

    def to(self, device_type: str) -> 'Model':
        """Move model to specified device"""
        self.device = Device(device_type)
        for layer in self.layers:
            layer.to(device_type)
        return self

    def add(self, layer: Layer):
        """Add layer with automatic shape adjustment"""
        if self.layers:
            prev_layer = self.layers[-1]

            # Try to auto-adjust layer shapes
            if hasattr(layer, 'input_size') and hasattr(prev_layer, 'output_size'):
                if layer.input_size != prev_layer.output_size:
                    print(f"Auto-adjusting layer input size from {layer.input_size} "
                          f"to {prev_layer.output_size}")
                    layer.input_size = prev_layer.output_size

        # Ensure layer is on same device as model
        layer.to(self.device.device_type)
        self.layers.append(layer)

    def _prepare_batch(self, X: Union[np.ndarray, List[np.ndarray], 'cp.ndarray'],
                       training: bool = True) -> Union[np.ndarray, 'cp.ndarray']:
        """Prepare batch data with proper shape handling"""
        # Handle list of sequences
        if isinstance(X, list):
            X = self.shape_handler.pad_sequences(X)
            X = self.device.to_device(X)
        else:
            X = self.device.to_device(X)

        # Ensure proper shape for first layer
        if self.layers:
            first_layer = self.layers[0]
            expected_shape = first_layer.get_expected_input_shape()
            X = self.shape_handler.auto_reshape(X, len(expected_shape))

        return X

    def _forward_pass(self, X: Union[np.ndarray, 'cp.ndarray'], training: bool = True) -> Union[np.ndarray, 'cp.ndarray']:
        """Forward pass with shape handling"""
        X = self._prepare_batch(X, training)
        output = X

        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training
            output = layer.forward(output)

        return output

    def _backward_pass(self, gradient: Union[np.ndarray, 'cp.ndarray']):
        """Backward pass through the network"""
        gradient = self.device.to_device(gradient)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def _update_params(self):
        """Update model parameters using optimizer"""
        for layer in self.layers:
            if hasattr(layer, 'get_trainable_params') and hasattr(layer, 'update_params'):
                params = layer.get_trainable_params()
                grads = layer.get_gradients()
                updated_params = self.optimizer.update(params, grads)
                layer.update_params(updated_params)

    def _get_weights(self) -> List:
        """Get current weights from all layers"""
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'get_weights'):
                weights.append(layer.get_weights())
        return weights

    def _set_weights(self, weights: List):
        """Set weights for all layers"""
        if len(weights) != len(self.layers):
            raise ValueError("Number of weight lists does not match number of layers")

        for layer, w in zip(self.layers, weights):
            if hasattr(layer, 'set_weights'):
                layer.set_weights(w)

    def compile(self, loss, optimizer):
        """Compile model with loss function and optimizer"""
        self.loss = loss
        self.optimizer = optimizer
        # Move loss and optimizer to current device if they support it
        if hasattr(self.loss, 'to'):
            self.loss.to(self.device.device_type)
        if hasattr(self.optimizer, 'to'):
            self.optimizer.to(self.device.device_type)

    def train(self,
              X: Union[np.ndarray, List[np.ndarray], 'cp.ndarray'],
              y: Union[np.ndarray, 'cp.ndarray'],
              epochs: int,
              batch_size: int,
              validation_data: Optional[Tuple[Union[np.ndarray, List[np.ndarray], 'cp.ndarray'],
                                           Union[np.ndarray, 'cp.ndarray']]] = None,
              early_stopping: bool = False,
              patience: int = 5,
              min_delta: float = 1e-4,
              verbose: int = 1) -> Dict[str, List]:
        """
        Train the model with support for variable batch sizes and partial batches

        Args:
            X: Input data (numpy array, cupy array, or list of arrays)
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Optional tuple of (X_val, y_val) for validation
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in loss to qualify as an improvement
            verbose: Verbosity level (0: silent, 1: progress)

        Returns:
            Dictionary containing training history
        """
        if not self.layers:
            raise ValueError("Model has no layers")
        if not self.loss or not self.optimizer:
            raise ValueError("Model must be compiled before training")

        # Move target data to device
        y = self.device.to_device(y)

        # Calculate total number of samples and batches
        n_samples = len(X)
        n_complete_batches = n_samples // batch_size
        has_partial_batch = n_samples % batch_size > 0

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [] if validation_data else None
        }

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(epochs):
            # Shuffle data
            if isinstance(X, list):
                indices = np.random.permutation(len(X))
                X_shuffled = [X[i] for i in indices]
                y_shuffled = y[indices]
            else:
                xp = self.device.xp
                indices = xp.random.permutation(len(X))
                X_shuffled = X[indices]
                y_shuffled = y[indices]

            # Training loop
            epoch_loss = 0
            n_batches = 0

            # Process all batches including the last partial batch
            for batch_idx in range(n_complete_batches + (1 if has_partial_batch else 0)):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                # Get current batch
                if isinstance(X_shuffled, list):
                    batch_X = X_shuffled[start_idx:end_idx]
                    batch_y = y_shuffled[start_idx:end_idx]
                else:
                    batch_X = X_shuffled[start_idx:end_idx]
                    batch_y = y_shuffled[start_idx:end_idx]

                # Forward pass (automatically handles variable batch sizes)
                predictions = self._forward_pass(batch_X, training=True)

                # Calculate loss (scale by actual batch size)
                current_batch_size = end_idx - start_idx
                batch_loss = self.loss.calculate(predictions, batch_y) * (current_batch_size / batch_size)
                epoch_loss += batch_loss

                # Backward pass
                grad = self.loss.derivative(predictions, batch_y)
                self._backward_pass(grad)

                # Update parameters
                self._update_params()
                n_batches += 1

            # Calculate average epoch loss
            epoch_loss /= n_batches
            history['train_loss'].append(float(self.device.to_cpu(epoch_loss)))

            # Validation phase
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val = self.device.to_device(y_val)

                # For validation, we can use a different batch size if needed
                val_predictions = self.predict(X_val)
                val_loss = self.loss.calculate(val_predictions, y_val)
                history['val_loss'].append(float(self.device.to_cpu(val_loss)))

                # Early stopping check
                if early_stopping:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_weights = self._get_weights()
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        if verbose:
                            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                        self._set_weights(best_weights)
                        break

            # Print progress
            if verbose:
                val_str = f", Val Loss: {val_loss:.4f}" if validation_data else ""
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}{val_str}")

        return history

    def predict(self, X: Union[np.ndarray, List[np.ndarray], 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Make predictions on new data"""
        return self._forward_pass(X, training=False)

    def summary(self):
        """Print model summary with shape information"""
        print("\nModel Summary:")
        print("=" * 50)
        print(f"Device: {self.device.device_type}")

        if not self.layers:
            print("Model is empty")
            return

        total_params = 0
        for i, layer in enumerate(self.layers):
            print(f"\nLayer {i}: {layer.__class__.__name__}")
            layer.print_shape_info()

            if hasattr(layer, 'get_weights'):
                weights = layer.get_weights()
                params = sum(w.size for w in weights if w is not None)
                total_params += params
                print(f"Parameters: {params:,}")

        print("\nTotal Parameters:", f"{total_params:,}")
        print("=" * 50)