import numpy as np
from typing import Dict, Optional, Union
from ..utils.lr_scheduler import LearningRateScheduler
from ..core.device import Device


class SGD:
    """
    SGD optimizer with GPU support, momentum and learning rate scheduling
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 lr_scheduler: Optional[LearningRateScheduler] = None):
        self.config = {
            'learning_rate': learning_rate,
            'momentum': momentum,
            'nesterov': nesterov
        }
        self.lr_scheduler = lr_scheduler
        self.velocities = {}
        self.device = Device('cpu')  # Default to CPU

    def to(self, device_type: str) -> 'SGD':
        """Move optimizer states to specified device"""
        self.device = Device(device_type)
        # Move velocities to device
        self.velocities = {
            k: self.device.to_device(v) for k, v in self.velocities.items()
        }
        return self

    def update(self,
               params: Dict[str, Union[np.ndarray, 'cp.ndarray']],
               grads: Dict[str, Union[np.ndarray, 'cp.ndarray']]) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """
        Update parameters using SGD with momentum and GPU support

        Args:
            params: Dictionary of parameters (CPU or GPU)
            grads: Dictionary of gradients (CPU or GPU)

        Returns:
            Dictionary of updated parameters on the same device
        """
        # Get current learning rate
        lr = (self.lr_scheduler.get_lr() if self.lr_scheduler
              else self.config['learning_rate'])

        # Initialize velocities if needed
        for param_name in params:
            if param_name not in self.velocities:
                self.velocities[param_name] = self.device.to_device(
                    self.device.xp.zeros_like(params[param_name])
                )

        # Update parameters
        updates = {}
        for param_name in params:
            # Ensure parameters and gradients are on correct device
            param = self.device.to_device(params[param_name])
            grad = self.device.to_device(grads[param_name])

            # Update velocity
            self.velocities[param_name] = (
                    self.config['momentum'] * self.velocities[param_name] -
                    lr * grad
            )

            if self.config['nesterov']:
                # Nesterov momentum update
                updates[param_name] = (
                        param +
                        self.config['momentum'] * self.velocities[param_name] -
                        lr * grad
                )
            else:
                # Standard momentum update
                updates[param_name] = (
                        param + self.velocities[param_name]
                )

        # Step learning rate scheduler
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return updates

    def get_config(self) -> dict:
        """Get optimizer configuration"""
        return {
            'class_name': self.__class__.__name__,
            'device': self.device.device_type,
            **self.config
        }

    def zero_grad(self):
        """Reset velocities"""
        self.velocities = {}