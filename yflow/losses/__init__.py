from .mse import MSELoss
from .cross_entropy import BinaryCrossEntropy, CrossEntropyLoss

__all__ = [
    'MSELoss',              # Mean Squared Error loss
    'BinaryCrossEntropy',   # Binary Cross Entropy loss
    'CrossEntropyLoss',     # Cross Entropy loss for language modeling
    'CELoss'                # Alias for CrossEntropyLoss
]

# Alias
CELoss = CrossEntropyLoss