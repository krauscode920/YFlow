# yflow/checkpoint/checkpoint_manager.py
"""
Automatic Checkpoint Management for YFlow.

Handles automatic saving, best model tracking, and checkpoint rotation.
"""

import os
from typing import Optional, Dict, Any
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    delete_old_checkpoints,
    get_checkpoint_info
)


class CheckpointManager:
    """
    Automatic checkpoint management for training loops.
    
    Features:
    - Automatic saving every N steps
    - Track and save best model
    - Automatic checkpoint rotation (keep only last N)
    - Easy checkpoint restoration
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        save_every_n_steps: Save checkpoint every N steps (default: 1000)
        keep_last_n: Keep only N most recent checkpoints (default: 5, 0 = keep all)
        track_best: Whether to track and save best model (default: True)
        best_metric: Metric to use for best model ('loss' or custom, default: 'loss')
        best_mode: 'min' or 'max' for best metric (default: 'min')
    
    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir='checkpoints/',
        ...     save_every_n_steps=1000,
        ...     keep_last_n=5,
        ...     track_best=True
        ... )
        >>> 
        >>> # In training loop
        >>> for step in range(10000):
        ...     loss = train_step()
        ...     
        ...     # Automatically saves if needed
        ...     manager.step(
        ...         model=model,
        ...         optimizer=optimizer,
        ...         step=step,
        ...         loss=loss
        ...     )
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_every_n_steps: int = 1000,
        keep_last_n: int = 5,
        track_best: bool = True,
        best_metric: str = 'loss',
        best_mode: str = 'min'
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n = keep_last_n
        self.track_best = track_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        
        # Best model tracking
        self.best_value = float('inf') if best_mode == 'min' else float('-inf')
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def step(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Check if checkpoint should be saved and save if needed.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            epoch: Current epoch
            loss: Current loss value
            metrics: Dictionary of metrics
            **kwargs: Additional metadata
        
        Returns:
            True if checkpoint was saved, False otherwise
        """
        saved = False
        
        # Save periodic checkpoint
        if step is not None and step > 0 and step % self.save_every_n_steps == 0:
            filepath = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.npz')
            save_checkpoint(
                filepath=filepath,
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch,
                loss=loss,
                metrics=metrics,
                **kwargs
            )
            saved = True
            
            # Cleanup old checkpoints
            if self.keep_last_n > 0:
                delete_old_checkpoints(
                    self.checkpoint_dir,
                    keep_last_n=self.keep_last_n,
                    pattern='checkpoint_step_*.npz'
                )
        
        # Save best model
        if self.track_best and self._is_best(loss, metrics):
            filepath = os.path.join(self.checkpoint_dir, 'best_model.npz')
            save_checkpoint(
                filepath=filepath,
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch,
                loss=loss,
                metrics=metrics,
                **kwargs
            )
            print(f"🏆 New best model! {self.best_metric} = {self.best_value:.6f}")
            saved = True
        
        return saved
    
    def save_epoch_checkpoint(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
        epoch: int = None,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Save a checkpoint at the end of an epoch.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch number
            step: Current training step
            loss: Current loss value
            metrics: Dictionary of metrics
            **kwargs: Additional metadata
        """
        filepath = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.npz')
        save_checkpoint(
            filepath=filepath,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            loss=loss,
            metrics=metrics,
            **kwargs
        )
        
        # Cleanup old epoch checkpoints
        if self.keep_last_n > 0:
            delete_old_checkpoints(
                self.checkpoint_dir,
                keep_last_n=self.keep_last_n,
                pattern='checkpoint_epoch_*.npz'
            )
    
    def restore_latest(self, model: Any, optimizer: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Restore the latest checkpoint.
        
        Args:
            model: Model to restore weights to
            optimizer: Optional optimizer to restore state to
        
        Returns:
            Checkpoint metadata if checkpoint found, None otherwise
        
        Example:
            >>> manager = CheckpointManager('checkpoints/')
            >>> metadata = manager.restore_latest(model, optimizer)
            >>> if metadata:
            ...     start_epoch = metadata['epoch']
            ...     start_step = metadata['step']
        """
        latest_checkpoint = get_latest_checkpoint(self.checkpoint_dir)
        
        if latest_checkpoint is None:
            print("⚠️  No checkpoint found to restore")
            return None
        
        return self.restore_checkpoint(latest_checkpoint, model, optimizer)
    
    def restore_best(self, model: Any, optimizer: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Restore the best model checkpoint.
        
        Args:
            model: Model to restore weights to
            optimizer: Optional optimizer to restore state to
        
        Returns:
            Checkpoint metadata if checkpoint found, None otherwise
        """
        best_checkpoint = os.path.join(self.checkpoint_dir, 'best_model.npz')
        
        if not os.path.exists(best_checkpoint):
            print("⚠️  No best model checkpoint found")
            return None
        
        return self.restore_checkpoint(best_checkpoint, model, optimizer)
    
    def restore_checkpoint(
        self,
        filepath: str,
        model: Any,
        optimizer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Restore a specific checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to restore weights to
            optimizer: Optional optimizer to restore state to
        
        Returns:
            Checkpoint metadata
        """
        checkpoint = load_checkpoint(filepath)
        
        # Restore model parameters
        if 'model_parameters' in checkpoint:
            if hasattr(model, 'set_parameters'):
                model.set_parameters(checkpoint['model_parameters'])
            else:
                raise AttributeError(
                    "Model must have a 'set_parameters()' method to restore checkpoint"
                )
        
        # Restore optimizer state
        if optimizer is not None and 'optimizer_state' in checkpoint:
            if hasattr(optimizer, 'set_state'):
                optimizer.set_state(checkpoint['optimizer_state'])
            else:
                raise AttributeError(
                    "Optimizer must have a 'set_state()' method to restore checkpoint"
                )
        
        # Return metadata
        return checkpoint.get('metadata', {})
    
    def _is_best(self, loss: Optional[float], metrics: Optional[Dict[str, Any]]) -> bool:
        """Check if current metrics represent the best model so far."""
        if not self.track_best:
            return False
        
        # Get the metric value to compare
        if self.best_metric == 'loss':
            current_value = loss
        elif metrics and self.best_metric in metrics:
            current_value = metrics[self.best_metric]
        else:
            return False
        
        if current_value is None:
            return False
        
        # Check if it's better
        is_better = False
        if self.best_mode == 'min':
            is_better = current_value < self.best_value
        elif self.best_mode == 'max':
            is_better = current_value > self.best_value
        
        if is_better:
            self.best_value = current_value
        
        return is_better
    
    def get_checkpoint_info(self, checkpoint_name: str = 'latest') -> Optional[Dict[str, Any]]:
        """
        Get metadata from a checkpoint without loading model weights.
        
        Args:
            checkpoint_name: 'latest', 'best', or specific checkpoint filename
        
        Returns:
            Dictionary of metadata or None if checkpoint not found
        """
        if checkpoint_name == 'latest':
            filepath = get_latest_checkpoint(self.checkpoint_dir)
        elif checkpoint_name == 'best':
            filepath = os.path.join(self.checkpoint_dir, 'best_model.npz')
        else:
            filepath = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        if filepath is None or not os.path.exists(filepath):
            return None
        
        return get_checkpoint_info(filepath)
    
    def __repr__(self) -> str:
        return (
            f"CheckpointManager(\n"
            f"  checkpoint_dir='{self.checkpoint_dir}',\n"
            f"  save_every_n_steps={self.save_every_n_steps},\n"
            f"  keep_last_n={self.keep_last_n},\n"
            f"  track_best={self.track_best},\n"
            f"  best_metric='{self.best_metric}',\n"
            f"  best_value={self.best_value:.6f}\n"
            f")"
        )
