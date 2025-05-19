# yflow/yformers/utils.py
"""
Utility functions for transformer models.

This module provides helper functions for working with transformer models,
including mask creation, positional encoding, and various other utilities.
All functions work with YFlow's device abstraction for seamless CPU/GPU support.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, List
from ..core.device import Device


def create_padding_mask(x, padding_idx=0, device_type='cpu'):
    """
    Create padding mask for transformer attention.

    Args:
        x: Input tensor of shape (batch_size, seq_len)
        padding_idx: Token ID that represents padding (default: 0)
        device_type: Device to use ('cpu' or 'gpu')

    Returns:
        Padding mask of shape (batch_size, 1, 1, seq_len)
    """
    device = Device(device_type)
    x = device.to_device(x)
    xp = device.xp

    # Mask where input tokens are padding
    mask = (x != padding_idx).astype(xp.float32)

    # Add broadcast dimensions for attention heads
    return mask[:, None, None, :]


def create_look_ahead_mask(seq_len, device_type='cpu'):
    """
    Create causal mask to prevent attention to future tokens.

    Args:
        seq_len: Length of sequence
        device_type: Device to use ('cpu' or 'gpu')

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    device = Device(device_type)
    xp = device.xp

    # Create lower triangular matrix (1s in lower triangle, 0s elsewhere)
    mask = xp.tril(xp.ones((seq_len, seq_len)))
    return mask


def create_combined_mask(x, padding_idx=0, device_type='cpu'):
    """
    Create combined padding and causal mask for transformer decoder.

    Args:
        x: Input tensor of shape (batch_size, seq_len)
        padding_idx: Token ID that represents padding (default: 0)
        device_type: Device to use ('cpu' or 'gpu')

    Returns:
        Combined mask of shape (batch_size, 1, seq_len, seq_len)
    """
    device = Device(device_type)
    x = device.to_device(x)
    xp = device.xp
    seq_len = x.shape[1]

    # Create padding mask (batch_size, 1, 1, seq_len)
    padding_mask = create_padding_mask(x, padding_idx=padding_idx, device_type=device_type)

    # Create causal mask (1, 1, seq_len, seq_len)
    look_ahead_mask = create_look_ahead_mask(seq_len, device_type=device_type)[None, None, :, :]

    # Combine masks: both conditions must be satisfied
    combined_mask = xp.minimum(padding_mask[:, :, :, None], look_ahead_mask)
    return combined_mask


def positional_encoding(max_len, d_model, device_type='cpu'):
    """
    Compute positional encoding as per the original transformer paper.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension
        device_type: Device to use ('cpu' or 'gpu')

    Returns:
        Positional encoding of shape (1, max_len, d_model)
    """
    device = Device(device_type)
    xp = device.xp

    # Initialize encoding matrix
    pe = xp.zeros((max_len, d_model))

    # Position and dimension indices
    position = xp.arange(0, max_len)[:, None]
    div_term = xp.exp(xp.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Apply sine to even indices
    pe[:, 0::2] = xp.sin(position * div_term)

    # Apply cosine to odd indices
    if d_model % 2 == 1:
        # Handle odd embedding dimensions
        pe[:, 1::2] = xp.cos(position * div_term)[:, :(d_model // 2)]
    else:
        pe[:, 1::2] = xp.cos(position * div_term)

    # Add batch dimension for broadcasting
    return pe[None, :, :]


def scaled_dot_product_attention(q, k, v, mask=None, device_type='cpu'):
    """
    Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (..., seq_len_q, depth)
        k: Key tensor of shape (..., seq_len_k, depth)
        v: Value tensor of shape (..., seq_len_v, depth)
        mask: Optional mask of shape (..., seq_len_q, seq_len_k)
        device_type: Device to use ('cpu' or 'gpu')

    Returns:
        Tuple of (attention_output, attention_weights)
    """
    device = Device(device_type)
    q = device.to_device(q)
    k = device.to_device(k)
    v = device.to_device(v)
    if mask is not None:
        mask = device.to_device(mask)

    xp = device.xp

    # Compute attention scores: (q*k^T) / sqrt(d_k)
    matmul_qk = xp.matmul(q, k.transpose(0, 1, 3, 2))

    # Scale by d_k
    d_k = q.shape[-1]
    scaled_attention_logits = matmul_qk / xp.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scaled_attention_logits += (1 - mask) * -1e9

    # Softmax over the last axis (seq_len_k)
    attention_weights = xp.softmax(scaled_attention_logits, axis=-1)

    # Compute output: attention_weights * v
    attention_output = xp.matmul(attention_weights, v)

    return attention_output, attention_weights


def prepare_transformer_inputs(src_tokens=None, tgt_tokens=None, device_type='cpu'):
    """
    Prepare inputs for transformer models, including masks.

    Args:
        src_tokens: Source tokens of shape (batch_size, src_seq_len)
        tgt_tokens: Target tokens of shape (batch_size, tgt_seq_len)
        device_type: Device to use ('cpu' or 'gpu')

    Returns:
        Dictionary with input tokens and appropriate masks
    """
    device = Device(device_type)
    inputs = {}

    if src_tokens is not None:
        src_tokens = device.to_device(src_tokens)
        inputs['src_tokens'] = src_tokens

        # Create padding mask for encoder
        inputs['src_mask'] = create_padding_mask(src_tokens, device_type=device_type)

    if tgt_tokens is not None:
        tgt_tokens = device.to_device(tgt_tokens)
        inputs['tgt_tokens'] = tgt_tokens

        # Create combined mask for decoder self-attention
        inputs['tgt_mask'] = create_combined_mask(tgt_tokens, device_type=device_type)

        if src_tokens is not None:
            # Create padding mask for encoder-decoder attention
            inputs['cross_mask'] = create_padding_mask(src_tokens, device_type=device_type)

    return inputs


def relative_position_encoding(max_len, d_model, device_type='cpu'):
    """
    Compute relative positional encoding.

    Used in more recent transformer variants like Transformer-XL.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension
        device_type: Device to use ('cpu' or 'gpu')

    Returns:
        Relative positional encoding of shape (2 * max_len - 1, d_model)
    """
    device = Device(device_type)
    xp = device.xp

    # Total positions needed (positive and negative positions)
    total_positions = 2 * max_len - 1
    positions = xp.arange(-max_len + 1, max_len)[:, None]

    # Compute using the same approach as standard positional encoding
    div_term = xp.exp(xp.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Initialize encoding matrix
    pe = xp.zeros((total_positions, d_model))

    # Apply sine to even indices
    pe[:, 0::2] = xp.sin(positions * div_term)

    # Apply cosine to odd indices
    if d_model % 2 == 1:
        # Handle odd embedding dimensions
        pe[:, 1::2] = xp.cos(positions * div_term)[:, :(d_model // 2)]
    else:
        pe[:, 1::2] = xp.cos(positions * div_term)

    return pe


def gelu_activation(x, device_type='cpu'):
    """
    Compute GELU activation function.

    Approximation of the GELU activation from "Gaussian Error Linear Units (GELUs)".

    Args:
        x: Input tensor
        device_type: Device to use ('cpu' or 'gpu')

    Returns:
        GELU activation output
    """
    device = Device(device_type)
    x = device.to_device(x)
    xp = device.xp

    # Approximate GELU activation
    return 0.5 * x * (1 + xp.tanh(xp.sqrt(2 / np.pi) * (x + 0.044715 * xp.power(x, 3))))


def apply_rotary_embeddings(q, k, cos, sin):
    """
    Apply rotary position embeddings to query and key tensors.

    Used in models like RoFormer and more recent transformers.

    Args:
        q: Query tensor of shape (..., seq_len, num_heads, head_dim)
        k: Key tensor of shape (..., seq_len, num_heads, head_dim)
        cos: Cosine part of rotary embedding
        sin: Sine part of rotary embedding

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Ensure all inputs are on the same device
    device = Device('cpu')
    q = device.to_device(q)
    k = device.to_device(k)
    cos = device.to_device(cos)
    sin = device.to_device(sin)
    xp = device.xp

    # Apply complex-valued rotary embeddings
    q_real, q_imag = q[..., ::2], q[..., 1::2]
    k_real, k_imag = k[..., ::2], k[..., 1::2]

    # Reshape cosine and sine for broadcasting
    cos = cos.reshape(*cos.shape, 1)
    sin = sin.reshape(*sin.shape, 1)

    # Complex multiply
    q_rotated_real = q_real * cos - q_imag * sin
    q_rotated_imag = q_real * sin + q_imag * cos
    k_rotated_real = k_real * cos - k_imag * sin
    k_rotated_imag = k_real * sin + k_imag * cos

    # Interleave real and imaginary parts
    q_rotated = xp.zeros_like(q)
    k_rotated = xp.zeros_like(k)
    q_rotated[..., ::2] = q_rotated_real
    q_rotated[..., 1::2] = q_rotated_imag
    k_rotated[..., ::2] = k_rotated_real
    k_rotated[..., 1::2] = k_rotated_imag

    return q_rotated, k_rotated


def generate_cosine_sinusoidal_position(seq_len, dim, device_type='cpu'):
    """
    Generate cosine and sine position embeddings for rotary embeddings.

    Args:
        seq_len: Sequence length
        dim: Dimension (must be even)
        device_type: Device to use ('cpu' or 'gpu')

    Returns:
        Tuple of (cos, sin) tensors
    """
    device = Device(device_type)
    xp = device.xp

    # Only for even dimensions
    assert dim % 2 == 0, "Dimension must be even for rotary embeddings"

    # Generate position indices
    positions = xp.arange(seq_len, dtype=xp.float32)

    # Generate dimension indices
    dims = xp.arange(0, dim, 2, dtype=xp.float32)

    # Compute angles
    inv_freq = 1.0 / (10000 ** (dims / dim))

    # Outer product to create a matrix of shape (seq_len, dim/2)
    sinusoid_inp = xp.outer(positions, inv_freq)

    # Compute cos and sin
    cos = xp.cos(sinusoid_inp)
    sin = xp.sin(sinusoid_inp)

    # Add an extra dimension for broadcasting with head dimension if needed
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    return cos, sin


def gradient_checkpoint(function, *args, **kwargs):
    """
    Basic implementation of gradient checkpointing.

    Use to trade computation for memory during backpropagation.

    Args:
        function: Function to checkpoint
        *args: Arguments to the function
        **kwargs: Keyword arguments to the function

    Returns:
        Function output
    """
    # In a full implementation, this would save activations and recompute
    # during backward pass. This is a placeholder for the concept.
    return function(*args, **kwargs)