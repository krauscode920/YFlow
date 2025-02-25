import numpy as np
from typing import List, Optional, Tuple, Union

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # Explicitly set cp to None if CuPy is not available


class ShapeHandler:
    """Comprehensive shape handling utility for YFlow with GPU support"""

    @staticmethod
    def auto_reshape(input_data: Union[np.ndarray, 'cp.ndarray'], target_ndim: int) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Automatically reshape input to target dimensions with robust GPU support.

        Args:
            input_data: Input array to reshape
            target_ndim: Desired number of dimensions

        Returns:
            Reshaped array with target number of dimensions
        """
        # Determine which array library to use
        if CUPY_AVAILABLE and isinstance(input_data, cp.ndarray):
            xp = cp
        else:
            xp = np

        current_ndim = input_data.ndim

        if current_ndim == target_ndim:
            return input_data

        if target_ndim == 3:  # For RNN/LSTM input
            if current_ndim == 2:
                return input_data.reshape(-1, 1, input_data.shape[-1])
            elif current_ndim == 1:
                return input_data.reshape(1, 1, -1)

        elif target_ndim == 2:  # For Dense layer input
            if current_ndim == 1:
                return input_data.reshape(1, -1)
            elif current_ndim == 3:
                return input_data.reshape(-1, input_data.shape[-1])

        raise ValueError(f"Cannot auto-reshape from {current_ndim}D to {target_ndim}D")

    @staticmethod
    def pad_sequences(sequences: List[Union[np.ndarray, 'cp.ndarray']],
                      max_length: Optional[int] = None,
                      padding: str = 'post',
                      truncating: str = 'post',
                      pad_value: float = 0.0) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Pad sequences to the same length with support for NumPy and CuPy.

        Args:
            sequences: List of input sequences
            max_length: Maximum length to pad/truncate to
            padding: 'pre' or 'post' padding
            truncating: 'pre' or 'post' truncating
            pad_value: Value to use for padding

        Returns:
            Padded array of sequences
        """
        if not sequences:
            return np.array([])

        # Determine which array library to use
        if CUPY_AVAILABLE and isinstance(sequences[0], cp.ndarray):
            xp = cp
        else:
            xp = np

        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        feature_shape = sequences[0].shape[1:]
        dtype = sequences[0].dtype

        padded_sequences = xp.full((len(sequences), max_length, *feature_shape),
                                   pad_value, dtype=dtype)

        for i, seq in enumerate(sequences):
            # Ensure sequence length
            if len(seq) > max_length:
                if truncating == 'pre':
                    seq = seq[-max_length:]
                else:
                    seq = seq[:max_length]

            # Padding
            if padding == 'post':
                padded_sequences[i, :len(seq)] = seq
            else:  # pre-padding
                padded_sequences[i, -len(seq):] = seq

        return padded_sequences

    @staticmethod
    def get_shape_str(shape: tuple) -> str:
        """
        Get human-readable shape description.

        Args:
            shape: Input shape tuple

        Returns:
            Formatted shape string
        """
        return ' × '.join(str(dim) if dim is not None else 'None'
                          for dim in shape)

    def __init__(self):
        """Initialize shape handler with default settings"""
        self.debug = False  # Print shape operations if True

    def log_shape_operation(self, operation: str,
                            input_shape: tuple,
                            output_shape: tuple):
        """
        Log shape transformation if debug is enabled.

        Args:
            operation: Description of shape operation
            input_shape: Input shape
            output_shape: Resulting output shape
        """
        if self.debug:
            print(f"{operation}: {self.get_shape_str(input_shape)} → "
                  f"{self.get_shape_str(output_shape)}")

    @staticmethod
    def ensure_same_device(arrays: List[Union[np.ndarray, 'cp.ndarray']]) -> List[Union[np.ndarray, 'cp.ndarray']]:
        """
        Ensure all arrays are on the same device.

        Args:
            arrays: List of input arrays

        Returns:
            List of arrays on the same device
        """
        if not arrays:
            return arrays

        # Determine target device from first array
        target_is_gpu = CUPY_AVAILABLE and isinstance(arrays[0], cp.ndarray)

        result = []
        for arr in arrays:
            is_gpu = CUPY_AVAILABLE and isinstance(arr, cp.ndarray)
            if is_gpu != target_is_gpu:
                if target_is_gpu and CUPY_AVAILABLE:
                    arr = cp.array(arr)
                else:
                    arr = cp.asnumpy(arr) if CUPY_AVAILABLE and isinstance(arr, cp.ndarray) else arr
            result.append(arr)

        return result