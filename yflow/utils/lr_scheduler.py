import numpy as np
from typing import Optional, Union, Dict


class LearningRateScheduler:
    """
    Learning rate scheduler with various decay options and GPU support

    Supported decay types:
    - step: Step decay every decay_steps
    - exponential: Continuous exponential decay
    - cosine: Cosine annealing
    - linear: Linear decay
    - custom: Custom decay function
    """

    def __init__(self,
                 initial_lr: float,
                 decay_type: str = 'step',
                 decay_rate: float = 0.1,
                 decay_steps: int = 1000,
                 min_lr: float = 0.0,
                 warmup_steps: int = 0):
        self.initial_lr = initial_lr
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.custom_decay_fn = None

        # Validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate scheduler parameters"""
        if self.initial_lr <= 0:
            raise ValueError("Initial learning rate must be positive")
        if self.decay_rate < 0:
            raise ValueError("Decay rate must be non-negative")
        if self.decay_steps <= 0:
            raise ValueError("Decay steps must be positive")
        if self.min_lr < 0:
            raise ValueError("Minimum learning rate must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("Warmup steps must be non-negative")

        valid_decay_types = ['step', 'exponential', 'cosine', 'linear', 'custom']
        if self.decay_type not in valid_decay_types:
            raise ValueError(f"Decay type must be one of {valid_decay_types}")

    def set_custom_decay_fn(self, decay_fn):
        """Set custom decay function"""
        self.decay_type = 'custom'
        self.custom_decay_fn = decay_fn

    def _apply_warmup(self, lr: float) -> float:
        """Apply linear warmup if configured"""
        if self.step_count < self.warmup_steps:
            return self.initial_lr * (self.step_count + 1) / self.warmup_steps
        return lr

    def get_lr(self) -> float:
        """Get current learning rate based on scheduler type"""
        if self.decay_type == 'custom' and self.custom_decay_fn is not None:
            lr = self.custom_decay_fn(self.initial_lr, self.step_count)
        else:
            lr = self._get_decay_lr()

        # Apply warmup and clip to minimum lr
        lr = self._apply_warmup(lr)
        return max(lr, self.min_lr)

    def _get_decay_lr(self) -> float:
        """Calculate learning rate based on decay type"""
        if self.step_count < self.warmup_steps:
            return self.initial_lr

        effective_step = self.step_count - self.warmup_steps

        if self.decay_type == 'step':
            return self.initial_lr * (self.decay_rate ** (effective_step // self.decay_steps))

        elif self.decay_type == 'exponential':
            return self.initial_lr * np.exp(-self.decay_rate * effective_step)

        elif self.decay_type == 'cosine':
            progress = np.clip(effective_step / self.decay_steps, 0, 1)
            return self.initial_lr * (1 + np.cos(np.pi * progress)) / 2

        elif self.decay_type == 'linear':
            progress = np.clip(effective_step / self.decay_steps, 0, 1)
            return self.initial_lr * (1 - progress)

        return self.initial_lr

    def step(self):
        """Increment step counter"""
        self.step_count += 1

    def reset(self):
        """Reset scheduler state"""
        self.step_count = 0

    def get_config(self) -> Dict:
        """Get scheduler configuration"""
        return {
            'initial_lr': self.initial_lr,
            'decay_type': self.decay_type,
            'decay_rate': self.decay_rate,
            'decay_steps': self.decay_steps,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'step_count': self.step_count
        }

    def state_dict(self) -> Dict:
        """Get scheduler state for saving"""
        return {
            'step_count': self.step_count,
            'config': self.get_config()
        }

    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state"""
        self.step_count = state_dict['step_count']
        config = state_dict['config']
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)