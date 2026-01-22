import numpy as np
from typing import Union
from ..core.device import Device


class BinaryCrossEntropy:
    """
    Binary Cross Entropy loss with GPU support and numerical stability
    """

    def __init__(self):
        self.device = Device('cpu')  # Default to CPU

    def to(self, device_type: str) -> 'BinaryCrossEntropy':
        """Move loss function to specified device"""
        self.device = Device(device_type)
        return self

    def calculate(self, y_pred: Union[np.ndarray, 'cp.ndarray'],
                  y_true: Union[np.ndarray, 'cp.ndarray']) -> float:
        """
        Calculate Binary Cross Entropy loss with GPU support

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

        # Clip predicted values to avoid log(0)
        eps = 1e-15
        y_pred = xp.clip(y_pred, eps, 1 - eps)

        # Calculate loss
        loss = -xp.mean(
            y_true * xp.log(y_pred) +
            (1 - y_true) * xp.log(1 - y_pred)
        )

        # Convert to float for any device
        return float(loss)

    def derivative(self, y_pred: Union[np.ndarray, 'cp.ndarray'],
                   y_true: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Calculate derivative of Binary Cross Entropy loss with GPU support

        Args:
            y_pred: Predicted values (CPU or GPU)
            y_true: True values (CPU or GPU)

        Returns:
            Loss gradient on same device as inputs
        """
        # Move inputs to correct device
        y_pred = self.device.to_device(y_pred)
        y_true = self.device.to_device(y_true)
        xp = self.device.xp

        # Clip for numerical stability
        eps = 1e-15
        y_pred = xp.clip(y_pred, eps, 1 - eps)

        # Calculate gradient
        return ((y_pred - y_true) /
                (y_pred * (1 - y_pred) + eps))