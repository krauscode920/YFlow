# yflow/yformers/__init__.py
"""
YFormers - Transformer architecture implementation for YFlow.
Built on top of YFlow's device abstraction and layer system.

This module provides transformer components including:
- Self-attention and multi-head attention
- Position embeddings (absolute and relative)
- Encoder blocks and full encoder stacks
- Decoder blocks and full decoder stacks
- Complete transformer models

All components follow YFlow's design principles, including
device abstraction (CPU/GPU support), gradient tracking,
and the ability to be integrated into larger YFlow models.
"""

# Import Attention Mechanisms
from .attention import (
    SelfAttention,
    MultiHeadAttention,
    CrossAttention
)

# Import Embeddings
from .embeddings import (
    TokenEmbedding,
    PositionalEncoding,
    PositionalEmbedding,
    LearnedPositionalEmbedding
)

# Import Encoder Components
from .encoder import (
    EncoderBlock,
    EncoderStack
)

# Import Decoder Components
from .decoder import (
    DecoderBlock,
    DecoderStack
)

# Import Complete Models
from .model import (
    TransformerModel,
    EncoderOnlyModel,
    DecoderOnlyModel
)

# Import Utility Functions
from .utils import (
    create_padding_mask,
    create_look_ahead_mask,
    create_combined_mask
)

# Version information
__version__ = '0.1.0'


def is_gpu_available():
    """
    Check if GPU is available for transformer operations.
    Delegates to YFlow's core GPU detection functionality.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    from ..core.device import Device
    device = Device('cpu')
    return device.is_gpu_available()


def get_device_info():
    """
    Get information about the available device.
    Useful for debugging transformer performance issues.

    Returns:
        dict: Device information dictionary including device type and memory stats
    """
    from ..core.device import Device
    device = Device('gpu' if is_gpu_available() else 'cpu')
    return device.get_memory_stats()


# List of all modules for easy access
__all__ = [
    # Attention mechanisms
    'SelfAttention',
    'MultiHeadAttention',
    'CrossAttention',

    # Embedding components
    'TokenEmbedding',
    'PositionalEncoding',
    'PositionalEmbedding',
    'LearnedPositionalEmbedding',

    # Encoder components
    'EncoderBlock',
    'EncoderStack',

    # Decoder components
    'DecoderBlock',
    'DecoderStack',

    # Complete models
    'TransformerModel',
    'EncoderOnlyModel',
    'DecoderOnlyModel',

    # Utility functions
    'create_padding_mask',
    'create_look_ahead_mask',
    'create_combined_mask',

    # Device utilities
    'is_gpu_available',
    'get_device_info'
]