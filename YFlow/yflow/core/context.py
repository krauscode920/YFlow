# yflow/core/context.py
import contextlib


class DeviceContext:
    """Global device context manager"""

    _current_device = None  # Static variable to track current device

    @classmethod
    def get_device(cls):
        """Get current device or create default"""
        if cls._current_device is None:
            from .device import Device
            cls._current_device = Device('cpu')
        return cls._current_device

    @classmethod
    def set_device(cls, device_type):
        """Set current device"""
        from .device import Device
        cls._current_device = Device(device_type)
        return cls._current_device

    @classmethod
    @contextlib.contextmanager
    def device(cls, device_type):
        """Context manager for temporary device change"""
        from .device import Device  # Import here to avoid circular imports
        old_device = cls._current_device
        cls._current_device = Device(device_type)
        try:
            yield cls._current_device
        finally:
            cls._current_device = old_device