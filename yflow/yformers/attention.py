# yflow/yformers/attention.py
"""
Attention mechanisms for transformer models.

This module implements various attention mechanisms used in transformer
architectures, including self-attention, multi-head attention, and cross-attention.
All implementations leverage YFlow's device abstraction for seamless CPU/GPU support.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from ..core.layer import Layer
from ..core.device import Device


class SelfAttention(Layer):
    """
    Self-attention mechanism as described in 'Attention Is All You Need'.

    Args:
        embed_dim (int): Size of input and output embeddings
        dropout (float): Dropout rate for attention weights
        mask_value (float): Value to use for masked positions in attention
    """

    def __init__(self,
                 embed_dim: int,
                 dropout: float = 0.0,
                 mask_value: float = -1e9):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout
        self.mask_value = mask_value
        self.training = True
        self.scale = 1.0 / np.sqrt(embed_dim)

        # Initialize weights - will be moved to correct device automatically
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize query, key, value weights"""
        xp = self.device.xp

        # Initialize Q, K, V weights with Xavier/Glorot uniform initialization
        limit = np.sqrt(6 / (2 * self.embed_dim))
        self.W_q = self.device.to_device(
            xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim))
        )
        self.W_k = self.device.to_device(
            xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim))
        )
        self.W_v = self.device.to_device(
            xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim))
        )

        # Initialize biases
        self.b_q = self.device.to_device(xp.zeros(self.embed_dim))
        self.b_k = self.device.to_device(xp.zeros(self.embed_dim))
        self.b_v = self.device.to_device(xp.zeros(self.embed_dim))

    def _apply_mask(self, attention_scores, mask):
        """Apply mask to attention scores"""
        if mask is not None:
            # Ensure mask is on the correct device
            mask = self.device.to_device(mask)
            # Apply mask by setting masked positions to a large negative value
            attention_scores = attention_scores + (1 - mask) * self.mask_value
        return attention_scores

    def _apply_dropout(self, x):
        """Apply dropout during training"""
        xp = self.device.xp
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            dropout_mask = xp.random.binomial(1, keep_prob, x.shape) / keep_prob
            return x * dropout_mask
        return x

    def forward(self, x, mask=None):
        """
        Forward pass for self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Move input to correct device
        x = self.device.to_device(x)
        xp = self.device.xp
        self.input = x  # Store for backward pass

        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V projections
        # Reshape for batch matrix multiplication
        q = xp.dot(x, self.W_q) + self.b_q  # (batch_size, seq_len, embed_dim)
        k = xp.dot(x, self.W_k) + self.b_k  # (batch_size, seq_len, embed_dim)
        v = xp.dot(x, self.W_v) + self.b_v  # (batch_size, seq_len, embed_dim)

        # Scaled dot-product attention
        # Transpose K for matrix multiplication
        scores = xp.matmul(q, k.transpose(0, 2, 1))  # (batch_size, seq_len, seq_len)
        scores = scores * self.scale

        # Apply mask if provided
        scores = self._apply_mask(scores, mask)

        # Softmax to get attention weights
        attention_weights = self._softmax(scores)

        # Apply dropout to attention weights
        attention_weights = self._apply_dropout(attention_weights)

        # Apply attention weights to values
        output = xp.matmul(attention_weights, v)  # (batch_size, seq_len, embed_dim)

        # Cache values for backward pass
        self.cache = {
            'q': q,
            'k': k,
            'v': v,
            'attention_weights': attention_weights,
            'mask': mask
        }

        return output

    def _softmax(self, x, axis=-1):
        """Compute softmax with numerical stability"""
        xp = self.device.xp
        x_max = xp.max(x, axis=axis, keepdims=True)
        e_x = xp.exp(x - x_max)
        return e_x / xp.sum(e_x, axis=axis, keepdims=True)

    def backward(self, output_gradient):
        """
        Backward pass for self-attention.

        Args:
            output_gradient: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        xp = self.device.xp
        output_gradient = self.device.to_device(output_gradient)

        q = self.cache['q']
        k = self.cache['k']
        v = self.cache['v']
        attention_weights = self.cache['attention_weights']
        mask = self.cache['mask']

        batch_size, seq_len, embed_dim = q.shape

        # Gradient with respect to values
        d_v = xp.matmul(attention_weights.transpose(0, 2, 1), output_gradient)

        # Gradient with respect to attention weights
        d_weights = xp.matmul(output_gradient, v.transpose(0, 2, 1))

        # Apply mask gradient if mask was used
        if mask is not None:
            mask = self.device.to_device(mask)
            d_weights = d_weights * mask

        # Gradient of softmax
        d_scores = d_weights * (attention_weights * (1 - attention_weights))

        # Apply scaling factor
        d_scores = d_scores * self.scale

        # Gradient with respect to Q
        d_q = xp.matmul(d_scores, k)

        # Gradient with respect to K
        d_k = xp.matmul(d_scores.transpose(0, 2, 1), q)

        # Gradient with respect to input x for Q, K, V projections
        d_x_q = xp.dot(d_q, self.W_q.T)
        d_x_k = xp.dot(d_k, self.W_k.T)
        d_x_v = xp.dot(d_v, self.W_v.T)

        # Sum the gradients for the input
        d_x = d_x_q + d_x_k + d_x_v

        # Gradients for weights
        self.dW_q = xp.dot(self.input.transpose(0, 2, 1), d_q).mean(axis=0)
        self.dW_k = xp.dot(self.input.transpose(0, 2, 1), d_k).mean(axis=0)
        self.dW_v = xp.dot(self.input.transpose(0, 2, 1), d_v).mean(axis=0)

        # Gradients for biases
        self.db_q = d_q.mean(axis=(0, 1))
        self.db_k = d_k.mean(axis=(0, 1))
        self.db_v = d_v.mean(axis=(0, 1))

        return d_x

    def get_trainable_params(self) -> Dict:
        """Get trainable parameters"""
        return {
            'W_q': self.W_q,
            'W_k': self.W_k,
            'W_v': self.W_v,
            'b_q': self.b_q,
            'b_k': self.b_k,
            'b_v': self.b_v
        }

    def get_gradients(self) -> Dict:
        """Get parameter gradients"""
        return {
            'W_q': self.dW_q,
            'W_k': self.dW_k,
            'W_v': self.dW_v,
            'b_q': self.db_q,
            'b_k': self.db_k,
            'b_v': self.db_v
        }

    def update_params(self, params: Dict):
        """Update layer parameters"""
        self.W_q = self.device.to_device(params['W_q'])
        self.W_k = self.device.to_device(params['W_k'])
        self.W_v = self.device.to_device(params['W_v'])
        self.b_q = self.device.to_device(params['b_q'])
        self.b_k = self.device.to_device(params['b_k'])
        self.b_v = self.device.to_device(params['b_v'])


class MultiHeadAttention(Layer):
    """
    Multi-head attention as described in 'Attention Is All You Need'.

    Args:
        embed_dim (int): Size of input and output embeddings
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate for attention weights
        mask_value (float): Value to use for masked positions in attention
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 mask_value: float = -1e9):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible "
                f"by number of heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout
        self.mask_value = mask_value
        self.training = True
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize query, key, value, and output projection weights"""
        xp = self.device.xp

        # Initialize Q, K, V weights
        limit = np.sqrt(6 / (2 * self.embed_dim))
        self.W_q = self.device.to_device(
            xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim))
        )
        self.W_k = self.device.to_device(
            xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim))
        )
        self.W_v = self.device.to_device(
            xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim))
        )

        # Output projection
        self.W_o = self.device.to_device(
            xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim))
        )

        # Initialize biases
        self.b_q = self.device.to_device(xp.zeros(self.embed_dim))
        self.b_k = self.device.to_device(xp.zeros(self.embed_dim))
        self.b_v = self.device.to_device(xp.zeros(self.embed_dim))
        self.b_o = self.device.to_device(xp.zeros(self.embed_dim))

    def _split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, head_dim)
        and reshape to (batch_size, num_heads, seq_len, head_dim)
        """
        xp = self.device.xp

        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        return xp.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x, batch_size):
        """
        Reverse of split_heads
        x shape: (batch_size, num_heads, seq_len, head_dim)
        returns: (batch_size, seq_len, embed_dim)
        """
        xp = self.device.xp

        # Transpose to (batch_size, seq_len, num_heads, head_dim)
        x = xp.transpose(x, (0, 2, 1, 3))

        # Reshape to (batch_size, seq_len, embed_dim)
        return x.reshape(batch_size, -1, self.embed_dim)

    def _apply_mask(self, attention_scores, mask):
        """Apply mask to attention scores"""
        if mask is not None:
            # Ensure mask is on the correct device
            mask = self.device.to_device(mask)

            # Expand mask for multi-head attention if needed
            if len(mask.shape) == 3:  # (batch_size, seq_len, seq_len)
                # Add head dimension: (batch_size, 1, seq_len, seq_len)
                mask = mask[:, None, :, :]

            # Apply mask by setting masked positions to a large negative value
            attention_scores = attention_scores + (1 - mask) * self.mask_value

        return attention_scores

    def _apply_dropout(self, x):
        """Apply dropout during training"""
        xp = self.device.xp
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            dropout_mask = xp.random.binomial(1, keep_prob, x.shape) / keep_prob
            return x * dropout_mask
        return x

    def _softmax(self, x, axis=-1):
        """Compute softmax with numerical stability"""
        xp = self.device.xp
        x_max = xp.max(x, axis=axis, keepdims=True)
        e_x = xp.exp(x - x_max)
        return e_x / xp.sum(e_x, axis=axis, keepdims=True)

    def forward(self, q, k=None, v=None, mask=None):
        """
        Forward pass for multi-head attention.

        For self-attention: call with q only (k and v default to q)
        For cross-attention: call with all q, k, v

        Args:
            q: Query tensor of shape (batch_size, seq_len_q, embed_dim)
            k: Key tensor of shape (batch_size, seq_len_k, embed_dim)
            v: Value tensor of shape (batch_size, seq_len_v, embed_dim)
            mask: Optional attention mask
                  For self-attention: shape (batch_size, seq_len_q, seq_len_q)
                  For cross-attention: shape (batch_size, seq_len_q, seq_len_k)

        Returns:
            Output tensor of shape (batch_size, seq_len_q, embed_dim)
        """
        # Handle self-attention case
        if k is None:
            k = q
        if v is None:
            v = q

        # Move inputs to correct device
        q = self.device.to_device(q)
        k = self.device.to_device(k)
        v = self.device.to_device(v)
        xp = self.device.xp

        self.input_q = q  # Store for backward pass
        self.input_k = k
        self.input_v = v

        batch_size = q.shape[0]

        # Linear projections
        q = xp.dot(q, self.W_q) + self.b_q  # (batch_size, seq_len_q, embed_dim)
        k = xp.dot(k, self.W_k) + self.b_k  # (batch_size, seq_len_k, embed_dim)
        v = xp.dot(v, self.W_v) + self.b_v  # (batch_size, seq_len_v, embed_dim)

        # Split heads
        q = self._split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, head_dim)
        k = self._split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, head_dim)
        v = self._split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, head_dim)

        # Scaled dot-product attention
        # Transpose K for matrix multiplication
        scores = xp.matmul(q, xp.transpose(k, (0, 1, 3, 2)))  # (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = scores * self.scale

        # Apply mask if provided
        scores = self._apply_mask(scores, mask)

        # Softmax to get attention weights
        attention_weights = self._softmax(scores, axis=-1)

        # Apply dropout to attention weights
        attention_weights = self._apply_dropout(attention_weights)

        # Apply attention weights to values
        context = xp.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, head_dim)

        # Combine heads
        context = self._combine_heads(context, batch_size)  # (batch_size, seq_len_q, embed_dim)

        # Final linear projection
        output = xp.dot(context, self.W_o) + self.b_o  # (batch_size, seq_len_q, embed_dim)

        # Cache values for backward pass
        self.cache = {
            'q_proj': q,
            'k_proj': k,
            'v_proj': v,
            'context': context,
            'attention_weights': attention_weights,
            'mask': mask
        }

        return output

    def backward(self, output_gradient):
        """
        Backward pass for multi-head attention.

        Args:
            output_gradient: Gradient from next layer of shape (batch_size, seq_len_q, embed_dim)

        Returns:
            Tuple of gradients with respect to q, k, v inputs
        """
        xp = self.device.xp
        output_gradient = self.device.to_device(output_gradient)

        q_proj = self.cache['q_proj']
        k_proj = self.cache['k_proj']
        v_proj = self.cache['v_proj']
        context = self.cache['context']
        attention_weights = self.cache['attention_weights']
        mask = self.cache['mask']

        batch_size = q_proj.shape[0]

        # Gradient through output projection
        d_context = xp.dot(output_gradient, self.W_o.T)
        self.dW_o = xp.dot(context.transpose(0, 2, 1), output_gradient).mean(axis=0)
        self.db_o = output_gradient.mean(axis=(0, 1))

        # Gradient through combine heads
        d_context = self._split_heads(d_context, batch_size)

        # Gradient through attention mechanism
        d_v = xp.matmul(xp.transpose(attention_weights, (0, 1, 3, 2)), d_context)

        # Gradient through attention weights
        d_scores = xp.matmul(d_context, xp.transpose(v_proj, (0, 1, 3, 2)))

        # Apply mask gradient if mask was used
        if mask is not None:
            mask = self.device.to_device(mask)
            if len(mask.shape) == 3:
                mask = mask[:, None, :, :]
            d_scores = d_scores * mask

        # Gradient of softmax
        d_scores = d_scores * (attention_weights * (1 - attention_weights))

        # Apply scaling factor
        d_scores = d_scores * self.scale

        # Gradient with respect to Q and K projections
        d_q = xp.matmul(d_scores, k_proj)
        d_k = xp.matmul(xp.transpose(d_scores, (0, 1, 3, 2)), q_proj)

        # Combine heads for gradients
        d_q = self._combine_heads(d_q, batch_size)
        d_k = self._combine_heads(d_k, batch_size)
        d_v = self._combine_heads(d_v, batch_size)

        # Gradient with respect to Q, K, V inputs
        d_input_q = xp.dot(d_q, self.W_q.T)
        d_input_k = xp.dot(d_k, self.W_k.T)
        d_input_v = xp.dot(d_v, self.W_v.T)

        # Gradients for weights
        self.dW_q = xp.dot(self.input_q.transpose(0, 2, 1), d_q).mean(axis=0)
        self.dW_k = xp.dot(self.input_k.transpose(0, 2, 1), d_k).mean(axis=0)
        self.dW_v = xp.dot(self.input_v.transpose(0, 2, 1), d_v).mean(axis=0)

        # Gradients for biases
        self.db_q = d_q.mean(axis=(0, 1))
        self.db_k = d_k.mean(axis=(0, 1))
        self.db_v = d_v.mean(axis=(0, 1))

        # Check if it's self-attention
        is_self_attention = (self.input_q is self.input_k) and (self.input_k is self.input_v)

        if is_self_attention:
            return d_input_q + d_input_k + d_input_v
        else:
            return d_input_q, d_input_k, d_input_v

    def get_trainable_params(self) -> Dict:
        """Get trainable parameters"""
        return {
            'W_q': self.W_q,
            'W_k': self.W_k,
            'W_v': self.W_v,
            'W_o': self.W_o,
            'b_q': self.b_q,
            'b_k': self.b_k,
            'b_v': self.b_v,
            'b_o': self.b_o
        }

    def get_gradients(self) -> Dict:
        """Get parameter gradients"""
        return {
            'W_q': self.dW_q,
            'W_k': self.dW_k,
            'W_v': self.dW_v,
            'W_o': self.dW_o,
            'b_q': self.db_q,
            'b_k': self.db_k,
            'b_v': self.db_v,
            'b_o': self.db_o
        }

    def update_params(self, params: Dict):
        """Update layer parameters"""
        for key, param in params.items():
            if hasattr(self, key):
                setattr(self, key, self.device.to_device(param))


# Alias for backward compatibility and easier usage
CrossAttention = MultiHeadAttention  # CrossAttention is just MultiHeadAttention with different q, k, v