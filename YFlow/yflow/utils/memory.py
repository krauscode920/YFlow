# yflow/utils/memory.py
class MemoryManager:
    """Helps manage device memory, especially for GPU operations"""

    def __init__(self, device):
        self.device = device
        self.tensors = {}  # Keep track of tensors

    def register(self, name, tensor):
        """Register a tensor to be tracked"""
        self.tensors[name] = tensor

    def free(self, name):
        """Free a specific tensor"""
        if name in self.tensors:
            del self.tensors[name]

    def clear_all(self):
        """Clear all registered tensors"""
        self.tensors.clear()
        self.device.clear_memory()

    def get_usage(self):
        """Get memory usage statistics"""
        if self.device.device_type != 'gpu':
            return {"available": None, "used": None}

        try:
            device = self.device.xp.cuda.Device()
            mem_info = device.mem_info
            return {
                "total": mem_info[0],
                "free": mem_info[1],
                "used": mem_info[0] - mem_info[1]
            }
        except:
            return {"available": None, "used": None}