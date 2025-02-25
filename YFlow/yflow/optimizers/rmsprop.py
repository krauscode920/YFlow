import numpy as np
from typing import Dict, Optional, Union
from ..utils.lr_scheduler import LearningRateScheduler
from ..core.device import Device


class RMSprop:
    """
    RMSprop optimizer with GPU support and improved functionality
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 rho: float = 0.9,
                 epsilon: float = 1e-8,
                 momentum: float = 0.0,
                 centered: bool = False,
                 gradient_clip_norm: Optional[float] = None,
                 lr_scheduler: Optional[LearningRateScheduler] = None):
        self.config = {
            'learning_rate': learning_rate,
            'rho': rho,
            'epsilon': epsilon,
            'momentum': momentum,
            'centered': centered,
            'gradient_clip_norm': gradient_clip_norm
        }
        self.lr_scheduler = lr_scheduler
        self.v = {}  # Moving average of squared gradients
        self.g = {} if centered else None  # Moving average of gradients (if centered)
        self.momentum_buffer = {} if momentum > 0 else None
        self.device = Device('cpu')  # Default to CPU

    def to(self, device_type: str) -> 'RMSprop':
        """Move optimizer states to specified device"""
        self.device = Device(device_type)
        # Move state buffers to device
        self.v = {k: self.device.to_device(v) for k, v in self.v.items()}
        if self.g is not None:
            self.g = {k: self.device.to_device(v) for k, v in self.g.items()}
        if self.momentum_buffer is not None:
            self.momentum_buffer = {k: self.device.to_device(v) for k, v in self.momentum_buffer.items()}
        return self

    def update(self,
               params: Dict[str, Union[np.ndarray, 'cp.ndarray']],
               grads: Dict[str, Union[np.ndarray, 'cp.ndarray']]) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """Update parameters using RMSprop with GPU support"""
        xp = self.device.xp

        # Get current learning rate
        lr = (self.lr_scheduler.get_lr() if self.lr_scheduler
              else self.config['learning_rate'])

        # Initialize accumulators if needed
        for param_name in params:
            if param_name not in self.v:
                param = self.device.to_device(params[param_name])
                self.v[param_name] = xp.zeros_like(param)
                if self.config['centered']:
                    self.g[param_name] = xp.zeros_like(param)
                if self.config['momentum'] > 0:
                    self.momentum_buffer[param_name] = xp.zeros_like(param)

        updates = {}
        for param_name in params:
            # Ensure parameters and gradients are on correct device
            param = self.device.to_device(params[param_name])
            grad = self.device.to_device(grads[param_name])

            # Update moving average of squared gradients
            self.v[param_name] = (
                    self.config['rho'] * self.v[param_name] +
                    (1 - self.config['rho']) * xp.square(grad)
            )

            if self.config['centered']:
                # Update moving average of gradients
                self.g[param_name] = (
                        self.config['rho'] * self.g[param_name] +
                        (1 - self.config['rho']) * grad
                )
                # Calculate update using centered RMSprop
                denom = xp.sqrt(
                    self.v[param_name] - xp.square(self.g[param_name]) +
                    self.config['epsilon']
                )
            else:
                # Standard RMSprop update
                denom = xp.sqrt(self.v[param_name] + self.config['epsilon'])

            # Apply momentum if used
            if self.config['momentum'] > 0:
                self.momentum_buffer[param_name] = (
                        self.config['momentum'] * self.momentum_buffer[param_name] +
                        lr * grad / denom
                )
                updates[param_name] = (
                        param - self.momentum_buffer[param_name]
                )
            else:
                updates[param_name] = param - lr * grad / denom

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
        """Reset optimizer states"""
        self.v = {}
        if self.config['centered']:
            self.g = {}
        if self.config['momentum'] > 0:
            self.momentum_buffer = {}