import numpy as np
from typing import Dict, Optional, Union
from ..utils.lr_scheduler import LearningRateScheduler
from ..core.device import Device

class Adam:
    """
    Adam optimizer with GPU support, learning rate scheduling and gradient clipping
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 amsgrad: bool = False,
                 gradient_clip_norm: Optional[float] = None,
                 lr_scheduler: Optional[LearningRateScheduler] = None):
        self.config = {
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'amsgrad': amsgrad,
            'gradient_clip_norm': gradient_clip_norm
        }
        self.lr_scheduler = lr_scheduler
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.v_hat = {} if amsgrad else None  # Amsgrad max second moments
        self.t = 0  # Time step
        self.device = Device('cpu')  # Default to CPU

    def to(self, device_type: str) -> 'Adam':
        """Move optimizer states to specified device"""
        self.device = Device(device_type)
        # Move moment estimates to device
        self.m = {k: self.device.to_device(v) for k, v in self.m.items()}
        self.v = {k: self.device.to_device(v) for k, v in self.v.items()}
        if self.v_hat is not None:
            self.v_hat = {k: self.device.to_device(v) for k, v in self.v_hat.items()}
        return self

    def _clip_gradients(self, grads: Dict[str, Union[np.ndarray, 'cp.ndarray']]) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """Clip gradients by global norm if specified with GPU support"""
        if self.config['gradient_clip_norm'] is None:
            return grads

        xp = self.device.xp
        # Calculate global norm
        global_norm = xp.sqrt(
            sum(xp.sum(xp.square(g)) for g in grads.values())
        )

        # Clip if necessary
        clip_norm = self.config['gradient_clip_norm']
        if global_norm > clip_norm:
            scale = clip_norm / (global_norm + self.config['epsilon'])
            return {k: v * scale for k, v in grads.items()}

        return grads

    def update(self,
               params: Dict[str, Union[np.ndarray, 'cp.ndarray']],
               grads: Dict[str, Union[np.ndarray, 'cp.ndarray']]) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """
        Update parameters using Adam optimization with GPU support

        Args:
            params: Dictionary of parameters (CPU or GPU)
            grads: Dictionary of gradients (CPU or GPU)

        Returns:
            Dictionary of updated parameters on the same device
        """
        xp = self.device.xp
        self.t += 1

        # Get current learning rate
        lr = (self.lr_scheduler.get_lr() if self.lr_scheduler
              else self.config['learning_rate'])

        # Ensure gradients are on correct device and clip if specified
        grads = {k: self.device.to_device(v) for k, v in grads.items()}
        grads = self._clip_gradients(grads)

        # Initialize momentum and velocity if needed
        for param_name in params:
            if param_name not in self.m:
                param = self.device.to_device(params[param_name])
                self.m[param_name] = xp.zeros_like(param)
                self.v[param_name] = xp.zeros_like(param)
                if self.config['amsgrad']:
                    self.v_hat[param_name] = xp.zeros_like(param)

        # Calculate bias correction terms
        beta1, beta2 = self.config['beta1'], self.config['beta2']
        bias_correction1 = 1 - beta1 ** self.t
        bias_correction2 = 1 - beta2 ** self.t

        # Update parameters
        updates = {}
        for param_name in params:
            # Ensure parameter is on correct device
            param = self.device.to_device(params[param_name])
            grad = grads[param_name]  # Already on correct device

            # Update moment estimates
            self.m[param_name] = (
                beta1 * self.m[param_name] +
                (1 - beta1) * grad
            )
            self.v[param_name] = (
                beta2 * self.v[param_name] +
                (1 - beta2) * xp.square(grad)
            )

            if self.config['amsgrad']:
                # Update max second moment estimate
                self.v_hat[param_name] = xp.maximum(
                    self.v_hat[param_name],
                    self.v[param_name]
                )
                # Use v_hat for parameter update
                v_corrected = self.v_hat[param_name]
            else:
                # Use regular v for parameter update
                v_corrected = self.v[param_name]

            # Compute update
            m_corrected = self.m[param_name] / bias_correction1
            v_corrected = v_corrected / bias_correction2

            updates[param_name] = (
                param -
                lr * m_corrected / (xp.sqrt(v_corrected) + self.config['epsilon'])
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
        """Reset optimizer states"""
        self.m = {}
        self.v = {}
        if self.config['amsgrad']:
            self.v_hat = {}
        self.t = 0