import numpy as np
from typing import Union
from ..core.device import Device


class MSELoss:
    """
    Mean Squared Error loss with GPU support and proper scaling
    """

    def __init__(self):
        self.device = Device('cpu')  # Default to CPU

    def to(self, device_type: str) -> 'MSELoss':
        """Move loss function to specified device"""
        self.device = Device(device_type)
        return self

    def calculate(self, y_pred: Union[np.ndarray, 'cp.ndarray'],
                  y_true: Union[np.ndarray, 'cp.ndarray']) -> float:
        """
        Calculate Mean Squared Error loss with GPU support

        Args:
            y_pred: Predicted values (CPU or GPU)
            y_true: True values (CPU or GPU)

        Returns:
            Computed loss as float
        """
        # Move inputs to correct device
        y_pred = self.device.to_device(y_pred)
        y_true = self.device.to_device(y_true)
        xp = self.device.xp

        # Calculate loss
        loss = xp.mean((y_pred - y_true) ** 2)

        # Convert to float for any device
        return float(loss)

    def derivative(self, y_pred: Union[np.ndarray, 'cp.ndarray'],
                   y_true: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Calculate derivative of Mean Squared Error loss with GPU support

        Args:
            y_pred: Predicted values (CPU or GPU)
            y_true: True values (CPU or GPU)

        Returns:
            Loss gradient on same device as inputs
        """
        # Move inputs to correct device
        y_pred = self.device.to_device(y_pred)
        y_true = self.device.to_device(y_true)

        # Calculate gradient
        return 2 * (y_pred - y_true) / y_pred.size