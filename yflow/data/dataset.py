# yflow/data/dataset.py
"""
Base Dataset class for YFlow.

Provides the foundation for creating custom datasets that work with FlowDL.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple


class Dataset(ABC):
    """
    Abstract base class for all datasets in YFlow.
    
    All custom datasets should inherit from this class and implement
    the required methods.
    
    Example:
        >>> class MyDataset(Dataset):
        ...     def __init__(self, data, labels):
        ...         self.data = data
        ...         self.labels = labels
        ...     
        ...     def __len__(self):
        ...         return len(self.data)
        ...     
        ...     def __getitem__(self, idx):
        ...         return self.data[idx], self.labels[idx]
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Get a sample from the dataset at the given index.
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            Sample data (can be a single item or tuple of items)
        """
        pass


class TensorDataset(Dataset):
    """
    Dataset wrapping tensors/arrays.
    
    Each sample is retrieved by indexing tensors along the first dimension.
    
    Args:
        *tensors: Tensors/arrays that have the same size in the first dimension
    
    Example:
        >>> import numpy as np
        >>> data = np.random.randn(100, 10)
        >>> labels = np.random.randint(0, 5, size=100)
        >>> dataset = TensorDataset(data, labels)
        >>> print(len(dataset))
        100
        >>> x, y = dataset[0]
        >>> print(x.shape, y)
        (10,) 3
    """
    
    def __init__(self, *tensors):
        if len(tensors) == 0:
            raise ValueError("At least one tensor must be provided")
        
        # Check all tensors have the same first dimension
        first_size = len(tensors[0])
        if not all(len(t) == first_size for t in tensors):
            raise ValueError("All tensors must have the same size in the first dimension")
        
        self.tensors = tensors
    
    def __len__(self) -> int:
        return len(self.tensors[0])
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a sample from all tensors at the given index.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of values from each tensor at the given index
        """
        if len(self.tensors) == 1:
            return self.tensors[0][idx]
        return tuple(tensor[idx] for tensor in self.tensors)
    
    def __repr__(self) -> str:
        return f"TensorDataset(num_tensors={len(self.tensors)}, size={len(self)})"
