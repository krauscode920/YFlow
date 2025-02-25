from .device import Device

class LayerConfig:
    """Base configuration class for layers with device support"""

    def __init__(self):
        self.device_type = 'cpu'  # Default device type

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config: dict) -> 'LayerConfig':
        instance = cls()
        for k, v in config.items():
            setattr(instance, k, v)
        return instance

    def to(self, device_type: str) -> 'LayerConfig':
        """Update device type"""
        self.device_type = device_type
        return self

class OptimizerConfig:
    """Base configuration class for optimizers with device support"""

    def __init__(self):
        self.device_type = 'cpu'  # Default device type

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config: dict) -> 'OptimizerConfig':
        instance = cls()
        for k, v in config.items():
            setattr(instance, k, v)
        return instance

    def to(self, device_type: str) -> 'OptimizerConfig':
        """Update device type"""
        self.device_type = device_type
        return self

def get_default_device() -> str:
    """Get default device type based on availability"""
    try:
        import cupy
        return 'gpu'
    except ImportError:
        return 'cpu'