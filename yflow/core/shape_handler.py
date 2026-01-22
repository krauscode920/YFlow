import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class ShapeHandler:
    """
    Enhanced shape handler for automatic tensor reshaping with 4D support.
    Handles transformations between 2D, 3D, and 4D tensors for various layer types.
    """
    
    def __init__(self, use_cupy=False):
        self.use_cupy = use_cupy and CUPY_AVAILABLE
        # Cache for shape information to aid reverse transformations
        self._last_batch_size = None
        self._last_seq_len = None
        self._num_heads = None
    
    def auto_reshape(self, input_data, target_ndim, **kwargs):
        """
        Automatically reshape input data to target dimensionality.
        
        Args:
            input_data: Input array to reshape
            target_ndim: Target number of dimensions (2, 3, or 4)
            **kwargs: Additional hints for reshaping:
                - batch_size: Batch size for 2D -> 3D
                - seq_len: Sequence length for 2D -> 3D
                - num_heads: Number of attention heads for 3D -> 4D
        
        Returns:
            Reshaped array with target_ndim dimensions
        """
        xp = cp.get_array_module(input_data) if self.use_cupy else np
        current_ndim = len(input_data.shape)
        
        # No reshaping needed
        if current_ndim == target_ndim:
            return input_data
        
        # Extract hints from kwargs
        batch_size = kwargs.get('batch_size', self._last_batch_size)
        seq_len = kwargs.get('seq_len', self._last_seq_len)
        num_heads = kwargs.get('num_heads', self._num_heads)
        
        # Update cache
        if batch_size is not None:
            self._last_batch_size = batch_size
        if seq_len is not None:
            self._last_seq_len = seq_len
        if num_heads is not None:
            self._num_heads = num_heads
        
        # Handle all possible transformations
        if target_ndim == 2:
            return self._to_2d(input_data, current_ndim)
        elif target_ndim == 3:
            return self._to_3d(input_data, current_ndim, batch_size, seq_len)
        elif target_ndim == 4:
            return self._to_4d(input_data, current_ndim, num_heads)
        else:
            raise ValueError(f"Target dimensionality {target_ndim} not supported. Use 2, 3, or 4.")
    
    def _to_2d(self, input_data, current_ndim):
        """Convert to 2D by flattening extra dimensions"""
        if current_ndim == 1:
            # (features,) -> (1, features)
            return input_data[None, :]
        elif current_ndim == 3:
            # (batch, seq, features) -> (batch*seq, features)
            batch_size, seq_len, features = input_data.shape
            self._last_batch_size = batch_size
            self._last_seq_len = seq_len
            return input_data.reshape(batch_size * seq_len, features)
        elif current_ndim == 4:
            # (batch, heads, seq, head_dim) -> (batch*heads*seq, head_dim)
            # or (batch, seq, heads, head_dim) -> (batch*seq, heads*head_dim)
            return input_data.reshape(-1, input_data.shape[-1])
        elif current_ndim > 4:
            # Flatten all but last dimension
            return input_data.reshape(-1, input_data.shape[-1])
        else:
            raise ValueError(f"Cannot convert {current_ndim}D to 2D")
    
    def _to_3d(self, input_data, current_ndim, batch_size=None, seq_len=None):
        """Convert to 3D"""
        if current_ndim == 1:
            # (features,) -> (1, 1, features)
            return input_data[None, None, :]
        elif current_ndim == 2:
            # (batch*seq, features) -> (batch, seq, features)
            if batch_size is None or seq_len is None:
                raise ValueError(
                    "Need batch_size and seq_len hints to reshape 2D to 3D. "
                    "Either provide them as kwargs or ensure they're cached from previous operations."
                )
            total_samples, features = input_data.shape
            if total_samples != batch_size * seq_len:
                # Try to infer
                if total_samples % seq_len == 0:
                    batch_size = total_samples // seq_len
                elif total_samples % batch_size == 0:
                    seq_len = total_samples // batch_size
                else:
                    raise ValueError(
                        f"Cannot reshape {total_samples} samples into batch_size={batch_size}, "
                        f"seq_len={seq_len}"
                    )
            return input_data.reshape(batch_size, seq_len, features)
        elif current_ndim == 4:
            # (batch, heads, seq, head_dim) -> (batch, seq, heads*head_dim)
            # Merge heads dimension into features
            batch, heads, seq, head_dim = input_data.shape
            # Transpose to (batch, seq, heads, head_dim) first
            input_data = input_data.transpose(0, 2, 1, 3)
            return input_data.reshape(batch, seq, heads * head_dim)
        elif current_ndim > 4:
            # Flatten middle dimensions
            return input_data.reshape(input_data.shape[0], -1, input_data.shape[-1])
        else:
            raise ValueError(f"Cannot convert {current_ndim}D to 3D")
    
    def _to_4d(self, input_data, current_ndim, num_heads=None):
        """Convert to 4D for multi-head attention"""
        if current_ndim == 2:
            # Not typically used, but handle for completeness
            # (samples, features) -> (1, num_heads, samples, head_dim)
            if num_heads is None:
                raise ValueError("Need num_heads hint to reshape 2D to 4D")
            samples, features = input_data.shape
            head_dim = features // num_heads
            if features % num_heads != 0:
                raise ValueError(
                    f"Features ({features}) must be divisible by num_heads ({num_heads})"
                )
            return input_data.reshape(1, samples, num_heads, head_dim).transpose(0, 2, 1, 3)
        elif current_ndim == 3:
            # (batch, seq, features) -> (batch, num_heads, seq, head_dim)
            if num_heads is None:
                raise ValueError("Need num_heads hint to reshape 3D to 4D")
            batch, seq, features = input_data.shape
            head_dim = features // num_heads
            if features % num_heads != 0:
                raise ValueError(
                    f"Features ({features}) must be divisible by num_heads ({num_heads})"
                )
            # Reshape to (batch, seq, num_heads, head_dim)
            reshaped = input_data.reshape(batch, seq, num_heads, head_dim)
            # Transpose to (batch, num_heads, seq, head_dim)
            return reshaped.transpose(0, 2, 1, 3)
        else:
            raise ValueError(f"Cannot convert {current_ndim}D to 4D")
    
    def pad_sequences(self, sequences, max_len=None, padding_value=0, dtype=np.float32):
        """
        Pad variable-length sequences to same length.
        
        Args:
            sequences: List of arrays with shape (seq_len, features) or (seq_len,)
            max_len: Maximum sequence length. If None, uses longest sequence.
            padding_value: Value to use for padding
            dtype: Data type for output array
        
        Returns:
            Padded array of shape (batch_size, max_len, features) or (batch_size, max_len)
        """
        xp = cp if self.use_cupy else np
        
        if not sequences:
            raise ValueError("Empty sequence list")
        
        # Determine max length
        if max_len is None:
            max_len = max(len(seq) for seq in sequences)
        
        # Determine if sequences have features dimension
        has_features = len(sequences[0].shape) > 1
        
        if has_features:
            features = sequences[0].shape[-1]
            padded = xp.full((len(sequences), max_len, features), padding_value, dtype=dtype)
        else:
            padded = xp.full((len(sequences), max_len), padding_value, dtype=dtype)
        
        # Fill in sequences
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            if seq_len > max_len:
                # Truncate if too long
                if has_features:
                    padded[i] = seq[:max_len]
                else:
                    padded[i] = seq[:max_len]
            else:
                # Pad if too short
                if has_features:
                    padded[i, :seq_len] = seq
                else:
                    padded[i, :seq_len] = seq
        
        return padded
    
    def ensure_same_device(self, *arrays):
        """
        Ensure all arrays are on the same device (CPU or GPU).
        
        Args:
            *arrays: Variable number of arrays
        
        Returns:
            Tuple of arrays all on the same device
        """
        if not arrays:
            return tuple()
        
        # Determine target device from first array
        first_array = arrays[0]
        if self.use_cupy and CUPY_AVAILABLE:
            xp = cp.get_array_module(first_array)
            target_device = 'gpu' if xp == cp else 'cpu'
        else:
            target_device = 'cpu'
            xp = np
        
        # Convert all arrays to target device
        result = []
        for arr in arrays:
            if arr is None:
                result.append(None)
                continue
            
            if target_device == 'gpu' and CUPY_AVAILABLE:
                if not isinstance(arr, cp.ndarray):
                    arr = cp.asarray(arr)
            else:
                if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
                    arr = cp.asnumpy(arr)
                elif not isinstance(arr, np.ndarray):
                    arr = np.asarray(arr)
            
            result.append(arr)
        
        return tuple(result)
    
    def get_shape_str(self, shape):
        """
        Get human-readable string representation of shape.
        
        Args:
            shape: Tuple representing tensor shape
        
        Returns:
            String representation with dimension names
        """
        if not shape:
            return "scalar"
        
        ndim = len(shape)
        
        if ndim == 1:
            return f"({shape[0]},) - 1D vector"
        elif ndim == 2:
            return f"({shape[0]}, {shape[1]}) - 2D (batch, features) or (samples, features)"
        elif ndim == 3:
            return f"({shape[0]}, {shape[1]}, {shape[2]}) - 3D (batch, seq_len, features)"
        elif ndim == 4:
            return (f"({shape[0]}, {shape[1]}, {shape[2]}, {shape[3]}) - "
                   f"4D (batch, num_heads, seq_len, head_dim)")
        else:
            return f"{shape} - {ndim}D tensor"
    
    def validate_shape(self, array, expected_ndim, name="input"):
        """
        Validate that array has expected number of dimensions.
        
        Args:
            array: Array to validate
            expected_ndim: Expected number of dimensions
            name: Name of the array for error messages
        
        Raises:
            ValueError: If shape doesn't match expectations
        """
        if array.ndim != expected_ndim:
            raise ValueError(
                f"{name} has {array.ndim} dimensions but expected {expected_ndim}. "
                f"Shape: {self.get_shape_str(array.shape)}"
            )
    
    def infer_batch_and_seq(self, shape_2d, total_samples):
        """
        Infer batch size and sequence length from 2D shape.
        Useful for reverse transformations.
        
        Args:
            shape_2d: Shape of 2D array (total_samples, features)
            total_samples: Total number of samples
        
        Returns:
            Tuple of (batch_size, seq_len)
        """
        if self._last_batch_size is not None and self._last_seq_len is not None:
            if self._last_batch_size * self._last_seq_len == total_samples:
                return self._last_batch_size, self._last_seq_len
        
        # Try common batch sizes
        common_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        for batch_size in common_batch_sizes:
            if total_samples % batch_size == 0:
                seq_len = total_samples // batch_size
                return batch_size, seq_len
        
        # Default to batch_size=1
        return 1, total_samples
