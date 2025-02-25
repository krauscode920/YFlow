import numpy as np
from typing import Union
from ..core.layer import Layer
from ..core.device import Device

class Dropout(Layer):
    """
    Dropout layer with GPU support and improved training/inference handling
    """

    def __init__(self, dropout_rate: float):
        super().__init__()
        if not 0 <= dropout_rate < 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, input_data: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Forward pass with GPU support and different behavior for training/inference

        Args:
            input_data: Input tensor (either on CPU or GPU)

        Returns:
            Output with dropout applied during training
        """
        # Move input to correct device
        input_data = self.device.to_device(input_data)
        xp = self.device.xp

        if self.training:
            # Generate dropout mask on the correct device
            self.mask = (xp.random.rand(*input_data.shape) > self.dropout_rate).astype(input_data.dtype)
            # Scale the output
            return input_data * self.mask / (1.0 - self.dropout_rate)
        return input_data

    def backward(self, output_gradient: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Backward pass with GPU support

        Args:
            output_gradient: Gradient from next layer

        Returns:
            Scaled gradient for previous layer
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)
        return output_gradient * self.mask / (1.0 - self.dropout_rate)

    def get_config(self) -> dict:
        """Get layer configuration"""
        return {
            'class_name': self.__class__.__name__,
            'dropout_rate': self.dropout_rate,
            'device': self.device.device_type
        }