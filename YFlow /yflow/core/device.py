import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # Explicitly set cp to None if CuPy is not available


class Device:
    def __init__(self, device_type='cpu'):
        """
        Initialize device with optional GPU support.

        Args:
            device_type (str): 'cpu' or 'gpu'
        """
        self.device_type = device_type
        if device_type == 'gpu' and not CUPY_AVAILABLE:
            raise RuntimeError("GPU device requested but CuPy is not available")

        # Select computational backend
        self.xp = cp if (device_type == 'gpu' and CUPY_AVAILABLE) else np

    def to_device(self, x):
        """
        Move array to the specified device.

        Args:
            x: Input array

        Returns:
            Array on the specified device
        """
        # If CuPy is not available, always return NumPy array
        if not CUPY_AVAILABLE:
            return np.asarray(x)

        # Convert based on current device type
        if isinstance(x, (np.ndarray, np.generic)):
            if self.device_type == 'gpu':
                return cp.asarray(x)
        elif CUPY_AVAILABLE and isinstance(x, (cp.ndarray, cp.generic)):
            if self.device_type == 'cpu':
                return cp.asnumpy(x)
        return x

    def to_cpu(self, x):
        """
        Ensure array is on CPU.

        Args:
            x: Input array

        Returns:
            CPU array
        """
        # If CuPy is not available, always return NumPy array
        if not CUPY_AVAILABLE:
            return np.asarray(x)

        # Convert CuPy array to NumPy if needed
        if CUPY_AVAILABLE and isinstance(x, (cp.ndarray, cp.generic)):
            return cp.asnumpy(x)

        return np.asarray(x)

    def is_gpu_available(self):
        """
        Check if GPU is available.

        Returns:
            bool: True if GPU is available, False otherwise
        """
        return CUPY_AVAILABLE and self.device_type == 'gpu'

    def __str__(self):
        """
        String representation of the device.

        Returns:
            str: Device description
        """
        return f"Device(type={self.device_type}, cupy_available={CUPY_AVAILABLE})"