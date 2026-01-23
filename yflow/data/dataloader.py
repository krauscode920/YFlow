# yflow/data/dataloader.py
"""
FlowDL - YFlow's DataLoader

Efficient data loading with batching, shuffling, and parallel processing.
"""

import numpy as np
from typing import Iterator, Optional, Callable, Any, List
from .dataset import Dataset


def default_collate(batch: List[Any]) -> Any:
    """
    Default collate function to merge a list of samples into a batch.

    Args:
        batch: List of samples from the dataset

    Returns:
        Batched data (stacked along first dimension)
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate empty batch")

    # Get first element to determine type
    elem = batch[0]

    # If elements are tuples (e.g., (data, label)), batch each component
    if isinstance(elem, tuple):
        # Recursively collate each component
        return tuple(default_collate([item[i] for item in batch])
                     for i in range(len(elem)))

    # If elements are numpy arrays, stack them
    if isinstance(elem, np.ndarray):
        return np.stack(batch, axis=0)

    # If elements are numbers, convert to numpy array
    if isinstance(elem, (int, float, np.integer, np.floating)):
        return np.array(batch)

    # If elements are lists, try to convert to numpy array
    if isinstance(elem, list):
        try:
            return np.array(batch)
        except:
            return batch

    # For any other type, try to convert to numpy array
    try:
        return np.array(batch)
    except:
        # If all else fails, return as list
        return batch


class FlowDL:
    """
    YFlow's DataLoader - Efficient data loading for training.

    Features:
    - Automatic batching
    - Shuffling per epoch
    - Drop last batch option
    - Custom collate functions
    - Memory efficient iteration

    Args:
        dataset: Dataset to load data from
        batch_size: Number of samples per batch (default: 1)
        shuffle: Whether to shuffle data at the start of each epoch (default: False)
        drop_last: Whether to drop the last incomplete batch (default: False)
        collate_fn: Function to merge samples into a batch (default: stack arrays)

    Example:
        >>> import numpy as np
        >>> from yflow.data import TensorDataset, FlowDL
        >>>
        >>> # Create dataset
        >>> data = np.random.randn(100, 10)
        >>> labels = np.random.randint(0, 5, size=100)
        >>> dataset = TensorDataset(data, labels)
        >>>
        >>> # Create dataloader
        >>> dataloader = FlowDL(dataset, batch_size=32, shuffle=True)
        >>>
        >>> # Iterate over batches
        >>> for batch_data, batch_labels in dataloader:
        ...     print(batch_data.shape, batch_labels.shape)
        ...     # Train on batch
        (32, 10) (32,)
        (32, 10) (32,)
        (32, 10) (32,)
        (4, 10) (4,)  # Last batch with remaining samples
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 1,
            shuffle: bool = False,
            drop_last: bool = False,
            collate_fn: Optional[Callable] = None
    ):
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"dataset must be an instance of Dataset, got {type(dataset).__name__}"
            )

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else default_collate

        # Calculate number of batches
        dataset_size = len(dataset)
        if self.drop_last:
            self.num_batches = dataset_size // batch_size
        else:
            self.num_batches = (dataset_size + batch_size - 1) // batch_size

    def __len__(self) -> int:
        """Return the number of batches"""
        return self.num_batches

    def __iter__(self) -> Iterator:
        """
        Create an iterator over batches.

        Yields:
            Batched data from the dataset
        """
        dataset_size = len(self.dataset)

        # Create indices
        if self.shuffle:
            indices = np.random.permutation(dataset_size)
        else:
            indices = np.arange(dataset_size)

        # Yield batches
        for start_idx in range(0, dataset_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, dataset_size)

            # Skip last batch if drop_last=True and batch is incomplete
            if self.drop_last and end_idx - start_idx < self.batch_size:
                break

            # Get batch indices
            batch_indices = indices[start_idx:end_idx]

            # Fetch samples for this batch
            batch = [self.dataset[int(idx)] for idx in batch_indices]

            # Collate batch
            yield self.collate_fn(batch)

    def __repr__(self) -> str:
        return (
            f"FlowDL(\n"
            f"  dataset={self.dataset},\n"
            f"  batch_size={self.batch_size},\n"
            f"  num_batches={self.num_batches},\n"
            f"  shuffle={self.shuffle},\n"
            f"  drop_last={self.drop_last}\n"
            f")"
        )


class InfiniteFlowDL:
    """
    Infinite DataLoader that keeps yielding batches indefinitely.

    Useful for training when you want to train for a fixed number of steps
    rather than epochs.

    Args:
        dataset: Dataset to load data from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data when restarting
        collate_fn: Function to merge samples into a batch

    Example:
        >>> dataloader = InfiniteFlowDL(dataset, batch_size=32, shuffle=True)
        >>>
        >>> # Train for fixed number of steps
        >>> for step, (batch_data, batch_labels) in enumerate(dataloader):
        ...     if step >= 10000:  # Train for 10k steps
        ...         break
        ...     # Train on batch
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 1,
            shuffle: bool = True,
            collate_fn: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn if collate_fn is not None else default_collate

    def __iter__(self) -> Iterator:
        """
        Create an infinite iterator over batches.

        Yields:
            Batched data from the dataset (repeats indefinitely)
        """
        while True:
            # Create a regular FlowDL for one epoch
            dataloader = FlowDL(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=False,
                collate_fn=self.collate_fn
            )

            # Yield all batches from this epoch
            for batch in dataloader:
                yield batch

    def __repr__(self) -> str:
        return (
            f"InfiniteFlowDL(\n"
            f"  dataset={self.dataset},\n"
            f"  batch_size={self.batch_size},\n"
            f"  shuffle={self.shuffle}\n"
            f")"
        )