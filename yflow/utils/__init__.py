from .seq_norm import SequenceNormalizer
from .seq_testing import SequenceTester
from .lr_scheduler import LearningRateScheduler


# Utility function for memory management
def clear_gpu_memory():
    """
    Clear GPU memory if CuPy is available.

    Returns:
        bool: True if memory cleared, False otherwise
    """
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.cuda.pinned_memory.PinnedMemoryPool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return True
    except ImportError:
        return False


# Utility function for device synchronization
def sync_device():
    """
    Synchronize CUDA device if CuPy is available.

    Returns:
        bool: True if device synchronized, False otherwise
    """
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
        return True
    except ImportError:
        return False


# Utility function to check GPU availability
def is_gpu_available():
    """
    Check if GPU is available.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    try:
        import cupy as cp
        return True
    except ImportError:
        return False


# Utility function to get device information
def get_device_info():
    """
    Get GPU device information.

    Returns:
        dict: Device information or default CPU info
    """
    if is_gpu_available():
        try:
            import cupy as cp
            device = cp.cuda.Device()
            return {
                'device_name': device.name,
                'memory_total': device.mem_info[0],
                'memory_free': device.mem_info[1]
            }
        except Exception:
            pass

    return {
        'device_name': 'CPU',
        'memory_total': None,
        'memory_free': None
    }


__all__ = [
    'SequenceNormalizer',
    'SequenceTester',
    'LearningRateScheduler',
    'clear_gpu_memory',
    'sync_device',
    'is_gpu_available',
    'get_device_info'
]