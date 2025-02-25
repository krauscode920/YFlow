import numpy as np
import logging
import importlib
import sys
import os
from typing import Union, Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO if os.environ.get('YFLOW_DEBUG') else logging.WARNING)
logger = logging.getLogger('yflow.device')


# Function for backward compatibility
def get_array_module(x):
    """Return appropriate array module (numpy or cupy) for the given array"""
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp
    except ImportError:
        pass
    return np


# Function to check if GPU is available
def is_gpu_available():
    """Check if GPU is available system-wide"""
    try:
        import cupy as cp
        try:
            cp.cuda.runtime.getDeviceCount()
            return True
        except:
            return False
    except ImportError:
        return False


class Device:
    """
    Enhanced device abstraction similar to PyTorch's approach.
    Supports automatic detection, installation of appropriate packages,
    and transparent handling of NumPy/CuPy arrays.
    """

    def __init__(self, device_type='cpu'):
        self.device_type = device_type
        self.xp = np  # Default to NumPy

        # For tracking memory usage
        self._registered_tensors = {}

        # Initialize device
        if device_type == 'gpu':
            self._setup_gpu()

    def _setup_gpu(self):
        """Setup GPU backend with appropriate error handling"""
        try:
            # First try to import cupy
            import cupy as cp
            self.xp = cp
            logger.info("Using CuPy backend for GPU operations")

            # Test GPU availability
            if not self.is_gpu_available():
                logger.warning("CuPy imported but no GPU detected. Falling back to CPU.")
                self.device_type = 'cpu'
                self.xp = np

        except ImportError:
            # Try to install appropriate CuPy version if on Colab
            if self._is_colab():
                self._install_compatible_cupy()
            else:
                logger.warning("GPU requested but CuPy not available. Falling back to CPU.")
                self.device_type = 'cpu'

    def _is_colab(self):
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False

    def _install_compatible_cupy(self):
        """Install CuPy version compatible with current CUDA"""
        try:
            import subprocess
            # Check CUDA version
            result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE)
            output = result.stdout.decode('utf-8')

            # Parse CUDA version
            import re
            match = re.search(r'release (\d+)\.(\d+)', output)
            if match:
                major = int(match.group(1))
                minor = int(match.group(2))

                # Install appropriate CuPy version
                if major == 11:
                    package = "cupy-cuda11x"
                elif major == 10:
                    package = "cupy-cuda10x"
                else:
                    package = f"cupy-cuda{major}0"

                logger.info(f"Installing {package} for CUDA {major}.{minor}")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

                # Try importing again
                import cupy as cp
                self.xp = cp
                logger.info(f"Successfully installed {package}")
            else:
                logger.warning("Could not determine CUDA version, falling back to CPU")
                self.device_type = 'cpu'
        except Exception as e:
            logger.warning(f"Error setting up GPU: {e}")
            self.device_type = 'cpu'

    def to_device(self, x):
        """
        Move data to current device (CPU or GPU)

        Args:
            x: Input data (array, list, dict, or scalar)

        Returns:
            Data on the current device
        """
        if x is None:
            return None

        if self.device_type == 'cpu':
            # CPU path
            if hasattr(x, 'get') and callable(x.get):  # Handle dict-like objects
                return {k: self.to_device(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [self.to_device(item) for item in x]
            elif hasattr(x, 'shape'):  # Array-like object
                try:
                    import cupy as cp
                    if isinstance(x, cp.ndarray):
                        # Convert from GPU to CPU
                        return cp.asnumpy(x)
                except ImportError:
                    pass
                return np.array(x)
            return x
        else:  # GPU
            if hasattr(x, 'get') and callable(x.get):  # Handle dict-like objects
                return {k: self.to_device(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [self.to_device(item) for item in x]
            elif hasattr(x, 'shape'):  # Array-like object
                if isinstance(x, self.xp.ndarray):  # Already on correct device
                    return x
                elif isinstance(x, np.ndarray):
                    # Move from CPU to GPU
                    return self.xp.array(x)
            elif np.isscalar(x):
                return x
            try:
                # Try to convert anything else to an array on the device
                return self.xp.array(x)
            except:
                logger.warning(f"Could not convert {type(x)} to device array")
                return x

    def to_cpu(self, x):
        """
        Move data to CPU from any device

        Args:
            x: Input data (array, list, dict, or scalar)

        Returns:
            Data on CPU (as NumPy arrays)
        """
        if x is None:
            return None

        if hasattr(x, 'get') and callable(x.get):  # Handle dict-like objects
            return {k: self.to_cpu(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self.to_cpu(item) for item in x]
        elif hasattr(x, 'shape'):  # Array-like object
            if self.device_type == 'gpu' and not isinstance(x, np.ndarray):
                try:
                    # Convert GPU array to CPU
                    return self.xp.asnumpy(x)
                except:
                    # If asnumpy fails, try generic conversion
                    return np.array(x)
            return np.array(x)
        return x

    def is_gpu_available(self):
        """Check if GPU is actually available"""
        if self.device_type != 'gpu':
            return False

        try:
            # Try accessing a GPU device
            self.xp.cuda.Device(0)
            # Get device properties to make sure it's working
            mem_info = self.xp.cuda.Device(0).mem_info
            return True
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return False

    def register_tensor(self, name: str, tensor) -> None:
        """
        Register a tensor to track memory usage

        Args:
            name: Name to identify the tensor
            tensor: The tensor to track
        """
        self._registered_tensors[name] = tensor

    def unregister_tensor(self, name: str) -> None:
        """
        Unregister a tracked tensor

        Args:
            name: Name of the tensor to stop tracking
        """
        if name in self._registered_tensors:
            del self._registered_tensors[name]

    def clear_memory(self):
        """Clear device memory if possible"""
        # Clear registered tensors
        self._registered_tensors.clear()

        # Clear GPU memory
        if self.device_type == 'gpu':
            try:
                mempool = self.xp.get_default_memory_pool()
                pinned_mempool = self.xp.cuda.pinned_memory.PinnedMemoryPool()
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()

                # Force garbage collection
                import gc
                gc.collect()
                return True
            except Exception as e:
                logger.warning(f"Memory clear failed: {e}")
                return False
        return True

    def sync(self):
        """Synchronize device operations"""
        if self.device_type == 'gpu':
            try:
                self.xp.cuda.Stream.null.synchronize()
                return True
            except Exception as e:
                logger.warning(f"Device sync failed: {e}")
                return False
        return True

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics

        Returns:
            Dictionary with memory information
        """
        stats = {
            'device_type': self.device_type,
            'registered_tensors': len(self._registered_tensors)
        }

        if self.device_type == 'gpu':
            try:
                device = self.xp.cuda.Device(0)
                mem_info = device.mem_info
                stats.update({
                    'total_bytes': mem_info[0],
                    'free_bytes': mem_info[1],
                    'used_bytes': mem_info[0] - mem_info[1],
                    'used_percent': 100 * (1 - mem_info[1] / mem_info[0]),
                    'device_name': device.name
                })
            except Exception as e:
                stats['error'] = str(e)

        return stats

    def __str__(self):
        """String representation with device info"""
        device_info = f"Device(type={self.device_type}"
        if self.device_type == 'gpu' and self.is_gpu_available():
            try:
                device = self.xp.cuda.Device(0)
                mem_info = device.mem_info
                device_info += f", name={device.name}, free={mem_info[1] / 1024 ** 3:.1f}GB"
            except:
                pass
        return device_info + ")"