# yflow/checkpoint/checkpoint.py
"""
Model Checkpointing for YFlow.

Save and load model weights, optimizer state, and training progress.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path


def save_checkpoint(
    filepath: str,
    model: Any,
    optimizer: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    loss: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Save a training checkpoint to disk.
    
    Saves model parameters, optimizer state, and training metadata to a .npz file.
    
    Args:
        filepath: Path where checkpoint will be saved (should end in .npz)
        model: Model instance with get_parameters() method
        optimizer: Optional optimizer instance with get_state() method
        epoch: Current epoch number
        step: Current training step
        loss: Current loss value
        metrics: Optional dictionary of metrics to save
        **kwargs: Additional metadata to save
    
    Example:
        >>> save_checkpoint(
        ...     'checkpoints/model_epoch_10.npz',
        ...     model=my_model,
        ...     optimizer=my_optimizer,
        ...     epoch=10,
        ...     step=5000,
        ...     loss=2.345
        ... )
    """
    # Ensure filepath ends with .npz
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint dictionary
    checkpoint = {}
    
    # Save model parameters
    if model is not None:
        if hasattr(model, 'get_parameters'):
            model_params = model.get_parameters()
            checkpoint['model_parameters'] = model_params
        else:
            raise AttributeError(
                "Model must have a 'get_parameters()' method that returns a dictionary of parameters"
            )
    
    # Save optimizer state
    if optimizer is not None:
        if hasattr(optimizer, 'get_state'):
            optimizer_state = optimizer.get_state()
            checkpoint['optimizer_state'] = optimizer_state
        else:
            raise AttributeError(
                "Optimizer must have a 'get_state()' method that returns a dictionary of state"
            )
    
    # Save training metadata
    metadata = {}
    if epoch is not None:
        metadata['epoch'] = epoch
    if step is not None:
        metadata['step'] = step
    if loss is not None:
        metadata['loss'] = loss
    if metrics is not None:
        metadata['metrics'] = metrics
    
    # Add any additional kwargs to metadata
    metadata.update(kwargs)
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    # Flatten nested dictionaries for numpy.savez
    flat_checkpoint = _flatten_dict(checkpoint)
    
    # Save to file
    np.savez_compressed(filepath, **flat_checkpoint)
    
    print(f"✅ Checkpoint saved to: {filepath}")


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """
    Load a checkpoint from disk.
    
    Args:
        filepath: Path to checkpoint file (.npz)
    
    Returns:
        Dictionary containing:
        - 'model_parameters': Model parameters (if saved)
        - 'optimizer_state': Optimizer state (if saved)
        - 'metadata': Training metadata (epoch, step, loss, etc.)
    
    Example:
        >>> checkpoint = load_checkpoint('checkpoints/model_epoch_10.npz')
        >>> model.set_parameters(checkpoint['model_parameters'])
        >>> optimizer.set_state(checkpoint['optimizer_state'])
        >>> start_epoch = checkpoint['metadata']['epoch']
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    # Load from file
    loaded = np.load(filepath, allow_pickle=True)
    
    # Reconstruct nested dictionary structure
    checkpoint = _unflatten_dict(loaded)
    
    print(f"✅ Checkpoint loaded from: {filepath}")
    
    return checkpoint


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
    """
    Flatten nested dictionary for numpy.savez.
    
    Converts:
        {'a': {'b': 1, 'c': 2}, 'd': 3}
    To:
        {'a/b': 1, 'a/c': 2, 'd': 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def _unflatten_dict(flat_dict: Any, sep: str = '/') -> Dict[str, Any]:
    """
    Reconstruct nested dictionary from flattened keys.
    
    Converts:
        {'a/b': 1, 'a/c': 2, 'd': 3}
    To:
        {'a': {'b': 1, 'c': 2}, 'd': 3}
    """
    result = {}
    
    for key in flat_dict.keys():
        value = flat_dict[key]
        
        # Handle numpy arrays and convert to proper type
        if isinstance(value, np.ndarray):
            # If it's a 0-d array, extract the scalar
            if value.ndim == 0:
                value = value.item()
        
        # Split key by separator
        parts = key.split(sep)
        
        # Navigate/create nested structure
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
    
    return result


def list_checkpoints(directory: str, pattern: str = "*.npz") -> List[str]:
    """
    List all checkpoint files in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match (default: "*.npz")
    
    Returns:
        List of checkpoint filepaths, sorted by modification time (newest first)
    
    Example:
        >>> checkpoints = list_checkpoints('checkpoints/')
        >>> print(checkpoints)
        ['checkpoints/model_epoch_15.npz', 'checkpoints/model_epoch_14.npz', ...]
    """
    if not os.path.exists(directory):
        return []
    
    path = Path(directory)
    checkpoint_files = list(path.glob(pattern))
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return [str(f) for f in checkpoint_files]


def get_latest_checkpoint(directory: str, pattern: str = "*.npz") -> Optional[str]:
    """
    Get the most recent checkpoint in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match (default: "*.npz")
    
    Returns:
        Path to most recent checkpoint, or None if no checkpoints found
    
    Example:
        >>> latest = get_latest_checkpoint('checkpoints/')
        >>> if latest:
        ...     checkpoint = load_checkpoint(latest)
    """
    checkpoints = list_checkpoints(directory, pattern)
    return checkpoints[0] if checkpoints else None


def delete_old_checkpoints(directory: str, keep_last_n: int = 5, pattern: str = "*.npz") -> None:
    """
    Delete old checkpoints, keeping only the N most recent.
    
    Args:
        directory: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        pattern: File pattern to match (default: "*.npz")
    
    Example:
        >>> # Keep only last 5 checkpoints, delete older ones
        >>> delete_old_checkpoints('checkpoints/', keep_last_n=5)
    """
    checkpoints = list_checkpoints(directory, pattern)
    
    if len(checkpoints) <= keep_last_n:
        return  # Nothing to delete
    
    # Delete old checkpoints
    for checkpoint in checkpoints[keep_last_n:]:
        try:
            os.remove(checkpoint)
            print(f"🗑️  Deleted old checkpoint: {checkpoint}")
        except OSError as e:
            print(f"⚠️  Failed to delete {checkpoint}: {e}")


def get_checkpoint_info(filepath: str) -> Dict[str, Any]:
    """
    Get metadata from a checkpoint without loading full model weights.
    
    Useful for quickly inspecting checkpoint information without loading
    large model parameters into memory.
    
    Args:
        filepath: Path to checkpoint file
    
    Returns:
        Dictionary containing metadata (epoch, step, loss, etc.)
    
    Example:
        >>> info = get_checkpoint_info('checkpoints/model_epoch_10.npz')
        >>> print(f"Epoch: {info['epoch']}, Loss: {info['loss']}")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    # Load only metadata keys (avoid loading large arrays)
    loaded = np.load(filepath, allow_pickle=True)
    
    # Extract metadata
    metadata = {}
    for key in loaded.keys():
        if key.startswith('metadata/'):
            value = loaded[key]
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            metadata[key.replace('metadata/', '')] = value
    
    return metadata
