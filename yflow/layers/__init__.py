from .lstm import YSTM
from .rnn import YQuence
from .dense import Dense
from .normalization import LayerNorm, BatchNormalization
from .dropout import Dropout
from .activations import (
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax,
    ELU,
    SELU,
    GELU
)


# Add utility function to check if GPU is available
def is_gpu_available():
    try:
        import cupy
        return True
    except ImportError:
        return False

# Add utility function to get device info
def get_device_info():
    if is_gpu_available():
        import cupy
        device = cupy.cuda.Device()
        return {
            'device_name': device.name,
            'memory_total': device.mem_info[0],
            'memory_free': device.mem_info[1]
        }
    return {'device_name': 'CPU', 'memory_total': None, 'memory_free': None}

__all__ = [
    'YSTM',          # LSTM implementation
    'YQuence',       # RNN implementation
    'Dense',         # Dense layer
    'LayerNorm',     # Layer normalization
    'BatchNormalization',  # Batch normalization
    'Dropout',       # Dropout layer
    'ReLU',          # Activation functions
    'LeakyReLU',
    'Sigmoid',
    'Tanh',
    'Softmax',
    'ELU',
    'SELU'
]