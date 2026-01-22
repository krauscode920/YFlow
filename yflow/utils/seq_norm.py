import numpy as np
from typing import Union, Dict, Optional, Tuple
from ..core.device import Device


class SequenceNormalizer:
    """
    Normalizer for sequence data with GPU support and multiple normalization methods.

    Methods:
    - minmax: Scale to [0,1] range
    - standard: Zero mean and unit variance
    - robust: Scale using percentiles
    - l2: L2 normalization

    Parameters:
        method (str): Normalization method ('minmax', 'standard', 'robust', 'l2')
        epsilon (float): Small constant for numerical stability
        device (str): Device to use ('cpu' or 'gpu')
    """

    def __init__(self,
                 method: str = 'minmax',
                 epsilon: float = 1e-8,
                 device: str = 'cpu'):
        self.method = method
        self.epsilon = epsilon
        self.device = Device(device)

        # Statistics
        self.max_val = None
        self.min_val = None
        self.mean = None
        self.std = None
        self.q1 = None  # For robust scaling
        self.q3 = None

        self._validate_params()

    def _validate_params(self):
        """Validate initialization parameters"""
        valid_methods = ['minmax', 'standard', 'robust', 'l2']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def fit(self, sequences: Union[np.ndarray, 'cp.ndarray']) -> None:
        """
        Compute scaling parameters from input sequences.

        Args:
            sequences: Input array of shape (batch_size, sequence_length, features)
                      or (batch_size, features)
        """
        sequences = self.device.to_device(sequences)
        xp = self.device.xp

        # Reshape to handle both 2D and 3D inputs
        all_values = sequences.reshape(-1)

        if self.method == 'minmax':
            self.max_val = xp.max(all_values)
            self.min_val = xp.min(all_values)

        elif self.method == 'standard':
            self.mean = xp.mean(all_values)
            self.std = xp.std(all_values)

        elif self.method == 'robust':
            # Move to CPU for percentile calculation if on GPU
            if isinstance(all_values, cp.ndarray):
                all_values = cp.asnumpy(all_values)
            self.q1 = np.percentile(all_values, 25)
            self.q3 = np.percentile(all_values, 75)

        # L2 normalization doesn't need fitting

    def transform(self, sequences: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Scale sequences according to the selected method.

        Args:
            sequences: Input array to be scaled

        Returns:
            Scaled array of same shape as input
        """
        sequences = self.device.to_device(sequences)
        xp = self.device.xp

        if self.method == 'minmax':
            if self.max_val is None or self.min_val is None:
                raise ValueError("Normalizer must be fitted before transform")
            return (sequences - self.min_val) / (self.max_val - self.min_val + self.epsilon)

        elif self.method == 'standard':
            if self.mean is None or self.std is None:
                raise ValueError("Normalizer must be fitted before transform")
            return (sequences - self.mean) / (self.std + self.epsilon)

        elif self.method == 'robust':
            if self.q1 is None or self.q3 is None:
                raise ValueError("Normalizer must be fitted before transform")
            iqr = self.q3 - self.q1 + self.epsilon
            return (sequences - self.q1) / iqr

        else:  # l2
            norm = xp.sqrt(xp.sum(sequences ** 2, axis=-1, keepdims=True))
            return sequences / (norm + self.epsilon)

    def fit_transform(self, sequences: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Combine fit and transform operations.

        Args:
            sequences: Input array

        Returns:
            Scaled array of same shape as input
        """
        self.fit(sequences)
        return self.transform(sequences)

    def inverse_transform(self, scaled_sequences: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Convert scaled sequences back to original range.

        Args:
            scaled_sequences: Array previously scaled by transform()

        Returns:
            Array in original scale
        """
        scaled_sequences = self.device.to_device(scaled_sequences)

        if self.method == 'minmax':
            if self.max_val is None or self.min_val is None:
                raise ValueError("Normalizer must be fitted before inverse_transform")
            return scaled_sequences * (self.max_val - self.min_val) + self.min_val

        elif self.method == 'standard':
            if self.mean is None or self.std is None:
                raise ValueError("Normalizer must be fitted before inverse_transform")
            return scaled_sequences * (self.std + self.epsilon) + self.mean

        elif self.method == 'robust':
            if self.q1 is None or self.q3 is None:
                raise ValueError("Normalizer must be fitted before inverse_transform")
            iqr = self.q3 - self.q1 + self.epsilon
            return scaled_sequences * iqr + self.q1

        else:  # l2
            return scaled_sequences  # No inverse transform for L2 normalization

    def to(self, device_type: str) -> 'SequenceNormalizer':
        """Move normalizer to specified device"""
        self.device = Device(device_type)
        # Move statistics to new device
        if self.max_val is not None:
            self.max_val = self.device.to_device(self.max_val)
        if self.min_val is not None:
            self.min_val = self.device.to_device(self.min_val)
        if self.mean is not None:
            self.mean = self.device.to_device(self.mean)
        if self.std is not None:
            self.std = self.device.to_device(self.std)
        return self

    def get_config(self) -> Dict:
        """Get normalizer configuration."""
        return {
            'method': self.method,
            'epsilon': self.epsilon,
            'device': self.device.device_type,
            'max_val': self.device.to_cpu(self.max_val) if self.max_val is not None else None,
            'min_val': self.device.to_cpu(self.min_val) if self.min_val is not None else None,
            'mean': self.device.to_cpu(self.mean) if self.mean is not None else None,
            'std': self.device.to_cpu(self.std) if self.std is not None else None,
            'q1': self.q1,
            'q3': self.q3
        }

    @classmethod
    def from_config(cls, config: Dict) -> 'SequenceNormalizer':
        """Create normalizer from configuration."""
        instance = cls(
            method=config['method'],
            epsilon=config['epsilon'],
            device=config['device']
        )
        instance.max_val = config['max_val']
        instance.min_val = config['min_val']
        instance.mean = config['mean']
        instance.std = config['std']
        instance.q1 = config['q1']
        instance.q3 = config['q3']
        return instance