# yflow/core/context.py
# yflow/core/context.py
import contextlib
import logging

logger = logging.getLogger('yflow.context')


class DeviceContext:
    """
    Global device context manager for managing GPU/CPU devices.

    This class provides:
    - Seamless device detection and initialization
    - Context management for device switching
    - Global device state tracking
    """

    _current_device = None  # Static variable to track current device

    @classmethod
    def get_device(cls):
        """
        Get current device or create a new default device.
        Automatically detects and selects GPU if available.

        Returns:
            Device: The current active device
        """
        if cls._current_device is None:
            from .device import Device
            # Initialize with CPU by default
            cls._current_device = Device('cpu')

            # Try importing cupy directly to ensure it's loaded
            try:
                import cupy
                logger.info("CuPy successfully imported in context")
            except ImportError:
                logger.info("CuPy not available, staying with CPU")

            # Now check if GPU is available and switch if it is
            if cls._current_device.is_gpu_available():
                logger.info("GPU detected, switching to GPU")
                cls._current_device = Device('gpu')

        return cls._current_device

    @classmethod
    def set_device(cls, device_type):
        """
        Set current global device.

        Args:
            device_type: 'cpu' or 'gpu'

        Returns:
            Device: The newly set device
        """
        from .device import Device
        old_device = cls._current_device
        cls._current_device = Device(device_type)

        # Log device change
        if old_device:
            logger.info(f"Device changed from {old_device.device_type} to {cls._current_device.device_type}")
        else:
            logger.info(f"Device set to {cls._current_device.device_type}")

        return cls._current_device

    @classmethod
    def reset_device(cls):
        """
        Reset device context, forcing re-detection on next get_device() call.
        Useful when GPU becomes available after initialization.
        """
        cls._current_device = None
        logger.info("Device context reset")

    @classmethod
    @contextlib.contextmanager
    def device(cls, device_type):
        """
        Context manager for temporary device change.

        Args:
            device_type: 'cpu' or 'gpu'

        Yields:
            Device: The temporary device for the context
        """
        from .device import Device  # Import here to avoid circular imports
        old_device = cls._current_device
        cls._current_device = Device(device_type)

        try:
            logger.info(f"Entered context with device: {device_type}")
            yield cls._current_device
        finally:
            cls._current_device = old_device
            logger.info(f"Exited context, restored device: {old_device.device_type if old_device else 'None'}")