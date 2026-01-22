import numpy as np
from typing import Dict, List, Union
from ..core.layer import Layer
from ..core.device import Device


class LayerNorm(Layer):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = None
        self.beta = None
        self.cache = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize parameters on the correct device"""
        xp = self.device.xp
        self.gamma = self.device.to_device(xp.ones(self.normalized_shape))
        self.beta = self.device.to_device(xp.zeros(self.normalized_shape))

    def forward(self, x: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Forward pass with GPU support"""
        xp = self.device.xp
        x = self.device.to_device(x)

        # Save input for backward pass
        self.cache['input'] = x

        # Calculate mean and variance along feature dimension
        mean = xp.mean(x, axis=-1, keepdims=True)
        var = xp.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / xp.sqrt(var + self.eps)

        # Scale and shift
        out = self.gamma * x_norm + self.beta

        # Cache variables for backward pass
        self.cache.update({
            'x_norm': x_norm,
            'var': var,
            'mean': mean,
            'gamma': self.gamma
        })

        return out

    def backward(self, dout: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Backward pass with GPU support"""
        xp = self.device.xp
        dout = self.device.to_device(dout)

        x = self.cache['input']
        x_norm = self.cache['x_norm']
        var = self.cache['var']
        mean = self.cache['mean']
        gamma = self.cache['gamma']
        N = x.shape[-1]

        # Gradients for gamma and beta
        self.dgamma = xp.sum(dout * x_norm, axis=(0, 1) if x.ndim == 3 else 0)
        self.dbeta = xp.sum(dout, axis=(0, 1) if x.ndim == 3 else 0)

        # Gradient with respect to x_norm
        dx_norm = dout * gamma

        # Gradient with respect to variance
        dvar = -0.5 * xp.sum(dx_norm * (x - mean) * (var + self.eps) ** (-1.5),
                             axis=-1, keepdims=True)

        # Gradient with respect to mean
        dmean = -xp.sum(dx_norm / xp.sqrt(var + self.eps), axis=-1, keepdims=True)
        dmean += -2 * dvar * xp.sum(x - mean, axis=-1, keepdims=True) / N

        # Gradient with respect to input x
        dx = dx_norm / xp.sqrt(var + self.eps)
        dx += 2 * dvar * (x - mean) / N
        dx += dmean / N

        return dx

    def get_trainable_params(self) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }

    def get_gradients(self) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        return {
            'gamma': self.dgamma,
            'beta': self.dbeta
        }

    def update_params(self, params: Dict[str, Union[np.ndarray, 'cp.ndarray']]):
        self.gamma = self.device.to_device(params['gamma'])
        self.beta = self.device.to_device(params['beta'])


class BatchNormalization(Layer):
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.cache = {}
        self.training = True
        self.gradients = {}

    def _initialize_parameters(self, input_shape: tuple):
        """Initialize parameters on the correct device"""
        xp = self.device.xp

        if self.gamma is None:
            self.gamma = self.device.to_device(xp.ones(input_shape[1:]))
        if self.beta is None:
            self.beta = self.device.to_device(xp.zeros(input_shape[1:]))
        if self.running_mean is None:
            self.running_mean = self.device.to_device(xp.zeros(input_shape[1:]))
        if self.running_var is None:
            self.running_var = self.device.to_device(xp.ones(input_shape[1:]))

    def forward(self, input_data: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Forward pass with GPU support"""
        xp = self.device.xp
        input_data = self.device.to_device(input_data)

        # Initialize parameters if needed
        self._initialize_parameters(input_data.shape)

        if self.training:
            # Calculate batch statistics
            batch_mean = xp.mean(input_data, axis=0)
            batch_var = xp.var(input_data, axis=0)

            # Update running statistics
            self.running_mean = (self.momentum * self.running_mean +
                                 (1.0 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var +
                                (1.0 - self.momentum) * batch_var)

            # Normalize using batch statistics
            normalized = ((input_data - batch_mean) /
                          xp.sqrt(batch_var + self.epsilon))

            # Cache values for backward pass
            self.cache.update({
                'input': input_data,
                'normalized': normalized,
                'batch_mean': batch_mean,
                'batch_var': batch_var,
                'gamma': self.gamma,
                'beta': self.beta
            })
        else:
            # Use running statistics for inference
            normalized = ((input_data - self.running_mean) /
                          xp.sqrt(self.running_var + self.epsilon))

        # Scale and shift
        return self.gamma * normalized + self.beta

    def backward(self, output_gradient: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Backward pass with GPU support"""
        xp = self.device.xp
        output_gradient = self.device.to_device(output_gradient)

        input_data = self.cache['input']
        normalized = self.cache['normalized']
        batch_mean = self.cache['batch_mean']
        batch_var = self.cache['batch_var']
        gamma = self.cache['gamma']
        N = input_data.shape[0]

        # Compute gradients for gamma and beta
        self.gradients['gamma'] = xp.sum(output_gradient * normalized, axis=0)
        self.gradients['beta'] = xp.sum(output_gradient, axis=0)

        # Gradient with respect to normalized input
        dnormalized = output_gradient * gamma

        # Gradient with respect to variance
        dvar = xp.sum(dnormalized * (input_data - batch_mean) *
                      -0.5 * (batch_var + self.epsilon) ** (-1.5), axis=0)

        # Gradient with respect to mean
        dmean = xp.sum(dnormalized * -1 / xp.sqrt(batch_var + self.epsilon), axis=0)
        dmean += dvar * xp.sum(-2 * (input_data - batch_mean), axis=0) / N

        # Gradient with respect to input
        dinput = (dnormalized / xp.sqrt(batch_var + self.epsilon) +
                  dvar * 2 * (input_data - batch_mean) / N +
                  dmean / N)

        return dinput

    def get_trainable_params(self) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }

    def get_gradients(self) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        return self.gradients

    def update_params(self, params: Dict[str, Union[np.ndarray, 'cp.ndarray']]):
        if 'gamma' in params:
            self.gamma = self.device.to_device(params['gamma'])
        if 'beta' in params:
            self.beta = self.device.to_device(params['beta'])

    def get_weights(self) -> List[Union[np.ndarray, 'cp.ndarray']]:
        return [
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var
        ]

    def set_weights(self, weights: List[Union[np.ndarray, 'cp.ndarray']]):
        self.gamma = self.device.to_device(weights[0])
        self.beta = self.device.to_device(weights[1])
        self.running_mean = self.device.to_device(weights[2])
        self.running_var = self.device.to_device(weights[3])

    def get_config(self) -> dict:
        return {
            'class_name': self.__class__.__name__,
            'epsilon': self.epsilon,
            'momentum': self.momentum
        }