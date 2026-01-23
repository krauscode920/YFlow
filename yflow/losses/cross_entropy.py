# yflow/losses/cross_entropy.py
"""
Cross-Entropy Loss for multi-class classification and language modeling.

This implementation provides:
- Numerically stable log-softmax computation
- Support for ignoring padding tokens
- GPU/CPU device abstraction
- Efficient gradient computation for backpropagation
"""

import numpy as np
from typing import Optional, Union
from ..core.device import Device


class CrossEntropyLoss:
    """
    Cross-Entropy Loss for next-token prediction in language models.

    Computes the cross-entropy loss between predicted logits and target tokens.
    Supports masking padding tokens and numerically stable computation.

    Args:
        ignore_index (int): Token ID to ignore in loss computation (typically padding token)
        reduction (str): How to reduce loss - 'mean', 'sum', or 'none'
        label_smoothing (float): Label smoothing factor (0.0 = no smoothing)

    Example:
        >>> loss_fn = CrossEntropyLoss(ignore_index=0)
        >>> logits = model(input_ids)  # Shape: (batch, seq_len, vocab_size)
        >>> loss = loss_fn.calculate(logits, target_ids)
        >>> grad = loss_fn.derivative(logits, target_ids)
        >>> model.backward(grad)
    """

    def __init__(self,
                 ignore_index: int = -100,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.device = Device('cpu')  # Default to CPU

        # Validate parameters
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

    def to(self, device_type: str) -> 'CrossEntropyLoss':
        """Move loss function to specified device"""
        self.device = Device(device_type)
        return self

    def _log_softmax(self, logits: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Compute log-softmax in a numerically stable way.

        Args:
            logits: Raw logits of shape (..., num_classes)

        Returns:
            Log probabilities of same shape as logits
        """
        xp = self.device.xp

        # Subtract max for numerical stability (log-sum-exp trick)
        logits_max = xp.max(logits, axis=-1, keepdims=True)
        logits_shifted = logits - logits_max

        # Compute log-softmax
        exp_logits = xp.exp(logits_shifted)
        sum_exp = xp.sum(exp_logits, axis=-1, keepdims=True)
        log_probs = logits_shifted - xp.log(sum_exp)

        return log_probs

    def _apply_label_smoothing(self,
                               targets: Union[np.ndarray, 'cp.ndarray'],
                               num_classes: int) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Apply label smoothing to target distribution.

        Converts hard targets to soft targets:
        - True class gets: 1 - label_smoothing
        - Other classes get: label_smoothing / (num_classes - 1)

        Args:
            targets: Target token IDs of shape (batch, seq_len)
            num_classes: Vocabulary size

        Returns:
            Smoothed target distribution of shape (batch, seq_len, num_classes)
        """
        if self.label_smoothing == 0.0:
            return targets

        xp = self.device.xp
        batch_size, seq_len = targets.shape

        # Create one-hot encoding
        one_hot = xp.zeros((batch_size, seq_len, num_classes))
        one_hot[xp.arange(batch_size)[:, None], xp.arange(seq_len), targets] = 1.0

        # Apply smoothing
        smoothed = one_hot * (1.0 - self.label_smoothing)
        smoothed += self.label_smoothing / num_classes

        return smoothed

    def calculate(self,
                  logits: Union[np.ndarray, 'cp.ndarray'],
                  targets: Union[np.ndarray, 'cp.ndarray'],
                  ignore_index: Optional[int] = None) -> float:
        """
        Calculate cross-entropy loss.

        Args:
            logits: Model predictions of shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs of shape (batch_size, seq_len)
            ignore_index: Override the default ignore_index for this call

        Returns:
            Scalar loss value
        """
        # Move inputs to correct device
        logits = self.device.to_device(logits)
        targets = self.device.to_device(targets)
        xp = self.device.xp

        # Use provided ignore_index or default
        ignore_idx = ignore_index if ignore_index is not None else self.ignore_index

        # Get dimensions
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape for easier computation
        # logits: (batch * seq_len, vocab_size)
        # targets: (batch * seq_len,)
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Compute log probabilities
        log_probs = self._log_softmax(logits_flat)

        # Create mask for valid (non-ignored) tokens
        mask = (targets_flat != ignore_idx).astype(xp.float32)
        num_valid = xp.sum(mask)

        if num_valid == 0:
            # All tokens are ignored
            return 0.0

        # Gather log probabilities for target tokens
        # For each position, get log_prob of the correct class
        batch_indices = xp.arange(logits_flat.shape[0])
        target_log_probs = log_probs[batch_indices, targets_flat]

        # Apply mask and compute loss
        masked_log_probs = target_log_probs * mask

        if self.reduction == 'none':
            # Return per-token loss
            loss = -masked_log_probs.reshape(batch_size, seq_len)
        elif self.reduction == 'sum':
            loss = -xp.sum(masked_log_probs)
        else:  # mean
            loss = -xp.sum(masked_log_probs) / num_valid

        # Cache for backward pass
        self.cache = {
            'log_probs': log_probs,
            'targets_flat': targets_flat,
            'mask': mask,
            'num_valid': num_valid,
            'original_shape': (batch_size, seq_len, vocab_size)
        }

        # Convert to float for scalar, return array for 'none' reduction
        if self.reduction == 'none':
            return loss
        return float(loss)

    def derivative(self,
                   logits: Union[np.ndarray, 'cp.ndarray'],
                   targets: Union[np.ndarray, 'cp.ndarray'],
                   ignore_index: Optional[int] = None) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Calculate gradient of cross-entropy loss with respect to logits.

        The gradient of cross-entropy with respect to logits is:
        grad = softmax(logits) - one_hot(targets)

        Args:
            logits: Model predictions of shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs of shape (batch_size, seq_len)
            ignore_index: Override the default ignore_index for this call

        Returns:
            Gradient with respect to logits, same shape as logits
        """
        # Move inputs to correct device
        logits = self.device.to_device(logits)
        targets = self.device.to_device(targets)
        xp = self.device.xp

        # Use provided ignore_index or default
        ignore_idx = ignore_index if ignore_index is not None else self.ignore_index

        # Get dimensions
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Compute softmax probabilities
        log_probs = self._log_softmax(logits_flat)
        probs = xp.exp(log_probs)

        # Create one-hot encoding of targets
        one_hot = xp.zeros_like(probs)
        batch_indices = xp.arange(logits_flat.shape[0])
        one_hot[batch_indices, targets_flat] = 1.0

        # Gradient: softmax - one_hot
        grad = probs - one_hot

        # Apply mask for ignored tokens
        mask = (targets_flat != ignore_idx).astype(xp.float32)
        grad = grad * mask[:, None]

        # Scale by reduction method
        if self.reduction == 'mean':
            num_valid = xp.sum(mask)
            if num_valid > 0:
                grad = grad / num_valid
        elif self.reduction == 'none':
            # Per-token loss, gradient already correct
            pass
        # For 'sum', gradient is already correct

        # Reshape back to original shape
        grad = grad.reshape(batch_size, seq_len, vocab_size)

        return grad

    def __call__(self,
                 logits: Union[np.ndarray, 'cp.ndarray'],
                 targets: Union[np.ndarray, 'cp.ndarray'],
                 ignore_index: Optional[int] = None) -> float:
        """
        Convenience method to calculate loss.
        Equivalent to calling calculate().
        """
        return self.calculate(logits, targets, ignore_index)

    def get_config(self) -> dict:
        """Get loss function configuration"""
        return {
            'class_name': self.__class__.__name__,
            'ignore_index': self.ignore_index,
            'reduction': self.reduction,
            'label_smoothing': self.label_smoothing,
            'device': self.device.device_type
        }


# Alias for compatibility
CELoss = CrossEntropyLoss

class BinaryCrossEntropy:
    """
    Binary Cross Entropy loss with GPU support and numerical stability
    """

    def __init__(self):
        self.device = Device('cpu')  # Default to CPU

    def to(self, device_type: str) -> 'BinaryCrossEntropy':
        """Move loss function to specified device"""
        self.device = Device(device_type)
        return self

    def calculate(self, y_pred: Union[np.ndarray, 'cp.ndarray'],
                  y_true: Union[np.ndarray, 'cp.ndarray']) -> float:
        """
        Calculate Binary Cross Entropy loss with GPU support

        Args:
            y_pred: Predicted values (CPU or GPU)
            y_true: True values (CPU or GPU)

        Returns:
            Computed loss as float
        """
        # Move inputs to correct device
        y_pred = self.device.to_device(y_pred)
        y_true = self.device.to_device(y_true)
        xp = self.device.xp

        # Clip predicted values to avoid log(0)
        eps = 1e-15
        y_pred = xp.clip(y_pred, eps, 1 - eps)

        # Calculate loss
        loss = -xp.mean(
            y_true * xp.log(y_pred) +
            (1 - y_true) * xp.log(1 - y_pred)
        )

        # Convert to float for any device
        return float(loss)

    def derivative(self, y_pred: Union[np.ndarray, 'cp.ndarray'],
                   y_true: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Calculate derivative of Binary Cross Entropy loss with GPU support

        Args:
            y_pred: Predicted values (CPU or GPU)
            y_true: True values (CPU or GPU)

        Returns:
            Loss gradient on same device as inputs
        """
        # Move inputs to correct device
        y_pred = self.device.to_device(y_pred)
        y_true = self.device.to_device(y_true)
        xp = self.device.xp

        # Clip for numerical stability
        eps = 1e-15
        y_pred = xp.clip(y_pred, eps, 1 - eps)

        # Calculate gradient
        return ((y_pred - y_true) /
                (y_pred * (1 - y_pred) + eps))