import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from ..core.layer import Layer, _set_global_seed
from ..core.device import Device

class Dense(Layer):
    """
    Dense (Fully Connected) layer with GPU support and improved initialization
    """

    def __init__(self,
                 output_size: int,
                 input_size: Optional[int] = None,
                 weight_init: str = 'glorot_uniform',
                 regularization: Optional[str] = None,
                 reg_strength: float = 0.01,
                 use_bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.weight_init = weight_init

        # Initialize weights if input_size is provided
        self.weights = None
        self.bias = None
        if self.input_size is not None:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using various initialization schemes"""
        xp = self.device.xp  # Use numpy or cupy based on device

        if self.weight_init.startswith('glorot'):
            limit = xp.sqrt(6 / (self.input_size + self.output_size))
            if self.weight_init == 'glorot_uniform':
                self.weights = self.device.to_device(
                    xp.random.uniform(-limit, limit, (self.input_size, self.output_size))
                )
            else:  # glorot_normal
                std = xp.sqrt(2 / (self.input_size + self.output_size))
                self.weights = self.device.to_device(
                    xp.random.normal(0, std, (self.input_size, self.output_size))
                )

        elif self.weight_init.startswith('he'):
            if self.weight_init == 'he_uniform':
                limit = xp.sqrt(6 / self.input_size)
                self.weights = self.device.to_device(
                    xp.random.uniform(-limit, limit, (self.input_size, self.output_size))
                )
            else:  # he_normal
                std = xp.sqrt(2 / self.input_size)
                self.weights = self.device.to_device(
                    xp.random.normal(0, std, (self.input_size, self.output_size))
                )

        elif self.weight_init.startswith('lecun'):
            if self.weight_init == 'lecun_uniform':
                limit = xp.sqrt(3 / self.input_size)
                self.weights = self.device.to_device(
                    xp.random.uniform(-limit, limit, (self.input_size, self.output_size))
                )
            else:  # lecun_normal
                std = xp.sqrt(1 / self.input_size)
                self.weights = self.device.to_device(
                    xp.random.normal(0, std, (self.input_size, self.output_size))
                )
        else:
            raise ValueError(f"Unknown initialization type: {self.weight_init}")

        if self.use_bias:
            self.bias = self.device.to_device(
                xp.zeros((1, self.output_size))
            )

    def get_expected_input_shape(self):
        """Get expected input shape with None for flexible dimensions"""
        if self.input_size is None:
            return (None, None)  # Both batch size and features are flexible
        return (None, self.input_size)  # Batch size is flexible, feature size is fixed

    # Fix for yflow/layers/dense.py
    # Replace the forward method in the Dense class:

    def forward(self, input_data):
        """Forward pass with automatic shape handling and GPU support"""

        # Initialize weights if this is the first forward pass
        if self.weights is None:
            if input_data.ndim == 3:
                # For 3D inputs (batch, seq, features), use the feature dimension
                self.input_size = input_data.shape[-1]
            else:
                self.input_size = input_data.shape[-1]
            self._initialize_weights()

        # Ensure input is on correct device
        self.input = self.device.to_device(input_data)

        # Handle 3D inputs (transformers need all timesteps, not just the last one)
        original_shape = None
        if self.input.ndim == 3:
            # Store original shape and reshape to 2D for matrix multiplication
            original_shape = self.input.shape  # (batch_size, seq_len, features)
            batch_size, seq_len, features = original_shape
            self.input = self.input.reshape(-1, features)  # (batch_size * seq_len, features)

        # Compute output using matrix multiplication
        output = self.device.xp.dot(self.input, self.weights)
        if self.use_bias:
            output = output + self.bias

        # Reshape back to 3D if input was 3D
        if original_shape is not None:
            batch_size, seq_len, _ = original_shape
            output = output.reshape(batch_size, seq_len, self.output_size)

        return output

    def backward(self, output_gradient):
        """Backward pass computing gradients with GPU support"""
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        # Calculate gradients
        self.weights_gradient = self.device.xp.dot(self.input.T, output_gradient)
        if self.use_bias:
            self.bias_gradient = self.device.xp.sum(output_gradient, axis=0, keepdims=True)

        # Add regularization gradient if specified
        if self.regularization:
            reg_gradient = self._regularization_gradient()
            self.weights_gradient += reg_gradient

        # Calculate gradient for previous layer
        input_gradient = self.device.xp.dot(output_gradient, self.weights.T)
        return input_gradient

    def _regularization_gradient(self):
        """Calculate regularization gradient with GPU support"""
        xp = self.device.xp
        if self.regularization == 'l2':
            return self.reg_strength * 2 * self.weights
        elif self.regularization == 'l1':
            return self.reg_strength * xp.sign(self.weights)
        return 0

    def _regularization_loss(self) -> float:
        """Calculate regularization loss with GPU support"""
        xp = self.device.xp
        if self.regularization == 'l2':
            return float(self.reg_strength * xp.sum(xp.square(self.weights)))
        elif self.regularization == 'l1':
            return float(self.reg_strength * xp.sum(xp.abs(self.weights)))
        return 0

    def get_weights(self) -> List:
        """Get current weights and bias"""
        weights = [self.weights]
        if self.use_bias:
            weights.append(self.bias)
        return weights

    def set_weights(self, weights: List):
        """Set weights and bias ensuring they're on the correct device"""
        self.weights = self.device.to_device(weights[0])
        if self.use_bias:
            self.bias = self.device.to_device(weights[1])

    def get_trainable_params(self) -> Dict:
        """Get trainable parameters"""
        params = {'weights': self.weights}
        if self.use_bias:
            params['bias'] = self.bias
        return params

    def get_gradients(self) -> Dict:
        """Get parameter gradients"""
        grads = {'weights': self.weights_gradient}
        if self.use_bias:
            grads['bias'] = self.bias_gradient
        return grads

    def update_params(self, params: Dict):
        """Update layer parameters ensuring they're on the correct device"""
        self.weights = self.device.to_device(params['weights'])
        if self.use_bias:
            self.bias = self.device.to_device(params['bias'])