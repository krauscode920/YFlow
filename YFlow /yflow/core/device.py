import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

def get_array_module(x):
    return cp.get_array_module(x) if CUPY_AVAILABLE else np

class Device:
    def __init__(self, device_type='cpu'):
        self.device_type = device_type
        if device_type == 'gpu' and not CUPY_AVAILABLE:
            print("Warning: GPU requested but CuPy is not available. Falling back to CPU.")
            self.device_type = 'cpu'
        self.xp = cp if (self.device_type == 'gpu' and CUPY_AVAILABLE) else np

    def to_device(self, x):
        if self.device_type == 'cpu':
            return np.array(x)
        else:
            import cupy as cp
            if isinstance(x, np.ndarray):
                return cp.array(x)
            return x

    def to_cpu(self, x):
        if not CUPY_AVAILABLE or isinstance(x, (np.ndarray, np.generic)):
            return np.asarray(x)
        if CUPY_AVAILABLE and isinstance(x, (cp.ndarray, cp.generic)):
            return cp.asnumpy(x)
        return np.asarray(x)

    def is_gpu_available(self):
        if not CUPY_AVAILABLE:
            return False
        try:
            cp.cuda.runtime.getDeviceCount()
            return True
        except:
            return False

    def __str__(self):
        return f"Device(type={self.device_type}, cupy_available={CUPY_AVAILABLE})"

def is_gpu_available():
    device = Device()
    return device.is_gpu_available()