import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import random
from .shape_handler import ShapeHandler
from .layer import Layer
from .device import Device, get_array_module

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
    def __init__(self, seed: Optional[int] = None):
        self.layers: List[Layer] = []
        self.loss = None
        self.optimizer = None
        self.shape_handler = ShapeHandler()
        self.seed = 42 if seed is None else seed
        self.device = Device('cpu')  # Default to CPU
        _set_global_seed(self.seed)

    def to(self, device_type: str) -> 'Model':
        self.device = Device(device_type)
        for layer in self.layers:
            layer.to(device_type)
        return self

    def add(self, layer: Layer):
        if self.layers:
            prev_layer = self.layers[-1]
            if hasattr(layer, 'input_size') and hasattr(prev_layer, 'output_size'):
                if layer.input_size != prev_layer.output_size:
                    print(f"Auto-adjusting layer input size from {layer.input_size} "
                          f"to {prev_layer.output_size}")
                    layer.input_size = prev_layer.output_size
        layer.to(self.device.device_type)
        self.layers.append(layer)

    def _prepare_batch(self, X: Union[np.ndarray, List[np.ndarray], 'cp.ndarray'],
                       training: bool = True) -> Union[np.ndarray, 'cp.ndarray']:
        xp = get_array_module(X)
        if isinstance(X, list):
            X = self.shape_handler.pad_sequences(X)
        X = self.device.to_device(X)
        if self.layers:
            first_layer = self.layers[0]
            expected_shape = first_layer.get_expected_input_shape()
            X = self.shape_handler.auto_reshape(X, len(expected_shape))
        return X

    def _forward_pass(self, X: Union[np.ndarray, 'cp.ndarray'], training: bool = True) -> Union[np.ndarray, 'cp.ndarray']:
        X = self._prepare_batch(X, training)
        output = X
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training
            output = layer.forward(output)
        return output

    def _backward_pass(self, gradient: Union[np.ndarray, 'cp.ndarray']):
        gradient = self.device.to_device(gradient)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def _update_params(self):
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
        self.loss = loss
        self.optimizer = optimizer
        if hasattr(self.loss, 'to'):
            self.loss.to(self.device.device_type)
        if hasattr(self.optimizer, 'to'):
            self.optimizer.to(self.device.device_type)

    def train(self, X, y, epochs, batch_size, validation_data=None,
              early_stopping=False, patience=5, min_delta=1e-4, verbose=1):
        if not self.layers or not self.loss or not self.optimizer:
            raise ValueError("Model must have layers and be compiled before training")

        xp = get_array_module(X)
        y = self.device.to_device(y)
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size

        history = {'train_loss': [], 'val_loss': [] if validation_data else None}
        best_val_loss, patience_counter, best_weights = float('inf'), 0, None

        for epoch in range(epochs):
            indices = xp.random.permutation(n_samples)
            X_shuffled = self.device.to_device(xp.array(X)[indices])
            y_shuffled = self.device.to_device(xp.array(y)[indices])

            epoch_loss = 0
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                batch_X = X_shuffled[start:end]
                batch_y = y_shuffled[start:end]

                predictions = self._forward_pass(batch_X, training=True)
                batch_loss = self.loss.calculate(predictions, batch_y)
                epoch_loss += batch_loss

                grad = self.loss.derivative(predictions, batch_y)
                self._backward_pass(grad)
                self._update_params()

            epoch_loss /= n_batches
            history['train_loss'].append(float(self.device.to_cpu(epoch_loss)))

            if validation_data:
                X_val, y_val = validation_data
                val_predictions = self.predict(X_val)
                val_loss = self.loss.calculate(val_predictions, y_val)
                history['val_loss'].append(float(self.device.to_cpu(val_loss)))

                if early_stopping:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss, patience_counter = val_loss, 0
                        best_weights = self._get_weights()
                    else:
                        patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                        self._set_weights(best_weights)
                        break

            if verbose:
                val_str = f", Val Loss: {val_loss:.4f}" if validation_data else ""
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}{val_str}")

        return history

    def predict(self, X):
        predictions = self._forward_pass(X, training=False)
        return self.device.to_cpu(predictions)

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