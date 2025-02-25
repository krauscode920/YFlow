import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import random
from .shape_handler import ShapeHandler
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


class Layer:
    """Base layer class with improved shape handling and device support"""

    def __init__(self):
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        self.training: bool = True
        self.shape_handler = ShapeHandler()
        self.cache = {}  # Storage for intermediate values
        self.device = Device('cpu')  # Default to CPU

    def to(self, device_type: str) -> 'Layer':
        """Move layer to specified device"""
        self.device = Device(device_type)
        # Move any cached data to device
        self.input = self.device.to_device(self.input) if self.input is not None else None
        self.output = self.device.to_device(self.output) if self.output is not None else None
        self.cache = {k: self.device.to_device(v) for k, v in self.cache.items()}
        return self

    def _validate_config(self):
        """Base configuration validation. Override in derived classes."""
        pass

    def _prepare_input(self, input_data: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Prepare input data with automatic shape handling."""
        # Move input to correct device
        input_data = self.device.to_device(input_data)

        # Get expected dimensions for this layer type
        from ..layers.lstm import YSTM
        from ..layers.rnn import YQuence
        expected_ndim = 3 if isinstance(self, (YQuence, YSTM)) else 2

        # Auto-reshape if needed
        if input_data.ndim != expected_ndim:
            input_data = self.shape_handler.auto_reshape(input_data, expected_ndim)

        return input_data

    def forward(self, input_data: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Forward pass with automatic shape handling"""
        input_data = self._prepare_input(input_data)
        self.input = input_data
        self.output = self._forward_impl(input_data)
        return self.output

    def _forward_impl(self, input_data: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Actual forward pass implementation - must be implemented by derived classes"""
        raise NotImplementedError

    def backward(self, output_gradient: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Backward pass - must be implemented by derived classes"""
        raise NotImplementedError

    def get_weights(self) -> List[Union[np.ndarray, 'cp.ndarray']]:
        """Get layer weights - override if layer has weights"""
        return []

    def set_weights(self, weights: List[Union[np.ndarray, 'cp.ndarray']]):
        """Set layer weights - override if layer has weights"""
        pass

    def get_config(self) -> dict:
        """Get layer configuration"""
        return {
            'class_name': self.__class__.__name__,
            'device': self.device.device_type
        }

    def get_trainable_params(self) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """Get trainable parameters - override if layer has trainable params"""
        return {}

    def get_gradients(self) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """Get parameter gradients - override if layer has trainable params"""
        return {}

    def update_params(self, params: Dict[str, Union[np.ndarray, 'cp.ndarray']]):
        """Update layer parameters - override if layer has trainable params"""
        pass

    def print_shape_info(self):
        """Print human-readable shape information"""
        input_shape = self.get_expected_input_shape()
        output_shape = self.compute_output_shape(input_shape)
        print(f"Layer: {self.__class__.__name__}")
        print(f"Device: {self.device.device_type}")
        print(f"Expected input shape: {self.shape_handler.get_shape_str(input_shape)}")
        print(f"Output shape: {self.shape_handler.get_shape_str(output_shape)}")

    def summary(self) -> Dict[str, Any]:
        """Get layer summary information"""
        params = self.get_trainable_params()
        n_params = sum(p.size for p in params.values()) if params else 0

        return {
            'name': self.__class__.__name__,
            'device': self.device.device_type,
            'input_shape': self.get_expected_input_shape(),
            'output_shape': self.compute_output_shape(self.get_expected_input_shape()),
            'n_params': n_params,
            'trainable': bool(params)
        }