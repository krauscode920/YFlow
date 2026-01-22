# Im using the file you gave me
# 6 #FINAL FIX - handles 1D, 2D, 3D, 4D masks correctly after squeeze
# yflow/yformers/attention.py
"""
Attention mechanisms for transformer models.
FIXED: Properly caches 4D split-head tensors and uses them in backward pass.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from ..core.layer import Layer
from ..core.device import Device


class SelfAttention(Layer):
    """Self-attention mechanism as described in 'Attention Is All You Need'."""

    def __init__(self, embed_dim: int, dropout: float = 0.0, mask_value: float = -1e9):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout
        self.mask_value = mask_value
        self.training = True
        self.scale = 1.0 / np.sqrt(embed_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        xp = self.device.xp
        limit = np.sqrt(6 / (2 * self.embed_dim))
        self.W_q = self.device.to_device(xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim)))
        self.W_k = self.device.to_device(xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim)))
        self.W_v = self.device.to_device(xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim)))
        self.b_q = self.device.to_device(xp.zeros(self.embed_dim))
        self.b_k = self.device.to_device(xp.zeros(self.embed_dim))
        self.b_v = self.device.to_device(xp.zeros(self.embed_dim))

    def _apply_mask(self, attention_scores, mask):
        if mask is not None:
            mask = self.device.to_device(mask)
            attention_scores = attention_scores + (1 - mask) * self.mask_value
        return attention_scores

    def _apply_dropout(self, x):
        xp = self.device.xp
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            dropout_mask = xp.random.binomial(1, keep_prob, x.shape) / keep_prob
            return x * dropout_mask
        return x

    def _softmax(self, x, axis=-1):
        xp = self.device.xp
        x_max = xp.max(x, axis=axis, keepdims=True)
        e_x = xp.exp(x - x_max)
        return e_x / xp.sum(e_x, axis=axis, keepdims=True)

    def forward(self, x, mask=None):
        x = self.device.to_device(x)
        xp = self.device.xp
        self.input = x

        q = xp.dot(x, self.W_q) + self.b_q
        k = xp.dot(x, self.W_k) + self.b_k
        v = xp.dot(x, self.W_v) + self.b_v

        scores = xp.matmul(q, k.transpose(0, 2, 1))
        scores = scores * self.scale
        scores = self._apply_mask(scores, mask)
        attention_weights = self._softmax(scores)
        attention_weights = self._apply_dropout(attention_weights)
        output = xp.matmul(attention_weights, v)

        self.cache = {'q': q, 'k': k, 'v': v, 'attention_weights': attention_weights, 'mask': mask}
        return output

    def backward(self, output_gradient):
        xp = self.device.xp
        output_gradient = self.device.to_device(output_gradient)
        q, k, v = self.cache['q'], self.cache['k'], self.cache['v']
        attention_weights, mask = self.cache['attention_weights'], self.cache['mask']

        d_v = xp.matmul(attention_weights.transpose(0, 2, 1), output_gradient)
        d_weights = xp.matmul(output_gradient, v.transpose(0, 2, 1))
        if mask is not None:
            d_weights = d_weights * self.device.to_device(mask)

        d_scores = d_weights * (attention_weights * (1 - attention_weights))
        d_scores = d_scores * self.scale
        d_q = xp.matmul(d_scores, k)
        d_k = xp.matmul(d_scores.transpose(0, 2, 1), q)

        d_x = xp.dot(d_q, self.W_q.T) + xp.dot(d_k, self.W_k.T) + xp.dot(d_v, self.W_v.T)
        self.dW_q = xp.dot(self.input.transpose(0, 2, 1), d_q).mean(axis=0)
        self.dW_k = xp.dot(self.input.transpose(0, 2, 1), d_k).mean(axis=0)
        self.dW_v = xp.dot(self.input.transpose(0, 2, 1), d_v).mean(axis=0)
        self.db_q = d_q.mean(axis=(0, 1))
        self.db_k = d_k.mean(axis=(0, 1))
        self.db_v = d_v.mean(axis=(0, 1))
        return d_x

    def get_trainable_params(self) -> Dict:
        return {'W_q': self.W_q, 'W_k': self.W_k, 'W_v': self.W_v,
                'b_q': self.b_q, 'b_k': self.b_k, 'b_v': self.b_v}

    def get_gradients(self) -> Dict:
        return {'W_q': self.dW_q, 'W_k': self.dW_k, 'W_v': self.dW_v,
                'b_q': self.db_q, 'b_k': self.db_k, 'b_v': self.db_v}

    def update_params(self, params: Dict):
        for key in ['W_q', 'W_k', 'W_v', 'b_q', 'b_k', 'b_v']:
            if key in params:
                setattr(self, key, self.device.to_device(params[key]))


class MultiHeadAttention(Layer):
    """Multi-head attention - CORRECTED to cache 4D split tensors."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, mask_value: float = -1e9):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout
        self.mask_value = mask_value
        self.training = True
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        xp = self.device.xp
        limit = np.sqrt(6 / (2 * self.embed_dim))
        self.W_q = self.device.to_device(xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim)))
        self.W_k = self.device.to_device(xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim)))
        self.W_v = self.device.to_device(xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim)))
        self.W_o = self.device.to_device(xp.random.uniform(-limit, limit, (self.embed_dim, self.embed_dim)))
        self.b_q = self.device.to_device(xp.zeros(self.embed_dim))
        self.b_k = self.device.to_device(xp.zeros(self.embed_dim))
        self.b_v = self.device.to_device(xp.zeros(self.embed_dim))
        self.b_o = self.device.to_device(xp.zeros(self.embed_dim))

    def _split_heads(self, x, batch_size):
        """Split: (batch, seq, embed) -> (batch, heads, seq, head_dim)"""
        xp = self.device.xp
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return xp.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x, batch_size):
        """Combine: (batch, heads, seq, head_dim) -> (batch, seq, embed)"""
        xp = self.device.xp
        while len(x.shape) > 4:
            for i in range(1, len(x.shape) - 3):
                if x.shape[i] == 1:
                    x = xp.squeeze(x, axis=i)
                    break
            else:
                if len(x.shape) == 5:
                    x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])
                break
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D tensor, got shape {x.shape}")
        x = xp.transpose(x, (0, 2, 1, 3))
        seq_len = x.shape[1]
        return x.reshape(batch_size, seq_len, self.embed_dim)

    def _apply_mask(self, attention_scores, mask):
        if mask is not None:
            mask = self.device.to_device(mask)
            xp = self.device.xp
            if len(mask.shape) == 2:
                mask = mask[None, None, :, :]
            elif len(mask.shape) == 3:
                mask = mask[:, None, :, :]
            elif len(mask.shape) == 5:
                mask = xp.squeeze(mask, axis=1) if mask.shape[1] == 1 else mask[:, 0, :, :, :]
            attention_scores = xp.where(mask > 0.5, attention_scores, self.mask_value)
        return attention_scores

    def _apply_dropout(self, x):
        xp = self.device.xp
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            return x * (xp.random.binomial(1, keep_prob, x.shape) / keep_prob)
        return x

    def _softmax(self, x, axis=-1):
        xp = self.device.xp
        x_max = xp.max(x, axis=axis, keepdims=True)
        e_x = xp.exp(x - x_max)
        return e_x / xp.sum(e_x, axis=axis, keepdims=True)

    def forward(self, q, k=None, v=None, mask=None):
        """Forward pass - caches 4D split tensors."""
        if k is None:
            k = q
        if v is None:
            v = q

        q = self.device.to_device(q)
        k = self.device.to_device(k)
        v = self.device.to_device(v)
        xp = self.device.xp

        self.input_q = q
        self.input_k = k
        self.input_v = v
        batch_size = q.shape[0]

        # Linear projections (3D)
        q_linear = xp.dot(q, self.W_q) + self.b_q
        k_linear = xp.dot(k, self.W_k) + self.b_k
        v_linear = xp.dot(v, self.W_v) + self.b_v

        # Split heads (4D)
        q_split = self._split_heads(q_linear, batch_size)
        k_split = self._split_heads(k_linear, batch_size)
        v_split = self._split_heads(v_linear, batch_size)

        # Attention computation
        scores = xp.matmul(q_split, xp.transpose(k_split, (0, 1, 3, 2)))
        scores = scores * self.scale
        scores = self._apply_mask(scores, mask)
        attention_weights = self._softmax(scores, axis=-1)
        attention_weights = self._apply_dropout(attention_weights)
        context_split = xp.matmul(attention_weights, v_split)

        # Combine heads (3D)
        context = self._combine_heads(context_split, batch_size)
        output = xp.dot(context, self.W_o) + self.b_o

        # Cache 4D split tensors!
        self.cache = {
            'q_split': q_split,
            'k_split': k_split,
            'v_split': v_split,
            'context': context,
            'attention_weights': attention_weights,
            'mask': mask
        }

        return output

    def backward(self, output_gradient):
        """Backward pass - uses cached 4D tensors."""
        xp = self.device.xp
        output_gradient = self.device.to_device(output_gradient)

        q_split = self.cache['q_split']
        k_split = self.cache['k_split']
        v_split = self.cache['v_split']
        context = self.cache['context']
        attention_weights = self.cache['attention_weights']
        mask = self.cache['mask']
        batch_size = q_split.shape[0]

        # Gradient through output projection
        d_context = xp.dot(output_gradient, self.W_o.T)
        self.dW_o = xp.dot(context.transpose(0, 2, 1), output_gradient).mean(axis=0)
        self.db_o = output_gradient.mean(axis=(0, 1))

        # Split gradient into heads
        d_context_split = self._split_heads(d_context, batch_size)

        # Attention gradients
        d_v_split = xp.matmul(attention_weights.transpose(0, 1, 3, 2), d_context_split)
        d_weights = xp.matmul(d_context_split, v_split.transpose(0, 1, 3, 2))

        if mask is not None:
            mask = self.device.to_device(mask)

            # Squeeze ALL size-1 dimensions at once
            mask = xp.squeeze(mask)

            # Now reshape to 4D based on what we have
            # Target: (batch, heads, seq_q, seq_k) where heads=1 initially
            if len(mask.shape) == 1:
                # (seq,) → (1, 1, 1, seq) - single sequence mask
                mask = mask[None, None, None, :]
            elif len(mask.shape) == 2:
                # Could be (batch, seq) or (seq_q, seq_k)
                # Check if square (causal mask) or not (padding mask)
                if mask.shape[0] == mask.shape[1]:
                    # (seq, seq) → (1, 1, seq, seq) - causal mask
                    mask = mask[None, None, :, :]
                else:
                    # (batch, seq) → (batch, 1, 1, seq) - padding mask
                    mask = mask[:, None, None, :]
            elif len(mask.shape) == 3:
                # (batch, seq_q, seq_k) → (batch, 1, seq_q, seq_k)
                mask = mask[:, None, :, :]
            # If already 4D, keep as is

            # Now ensure heads dimension broadcasts correctly
            if len(mask.shape) == 4 and mask.shape[1] == 1 and d_weights.shape[1] != 1:
                # Broadcast across heads
                mask = xp.broadcast_to(mask, d_weights.shape)

            d_weights = d_weights * mask

        # Softmax gradient
        sum_term = xp.sum(d_weights * attention_weights, axis=-1, keepdims=True)
        d_scores = attention_weights * (d_weights - sum_term)
        d_scores = d_scores * self.scale

        # Q and K gradients
        d_q_split = xp.matmul(d_scores, k_split)
        d_k_split = xp.matmul(d_scores.transpose(0, 1, 3, 2), q_split)

        # Combine heads
        d_q_3d = self._combine_heads(d_q_split, batch_size)
        d_k_3d = self._combine_heads(d_k_split, batch_size)
        d_v_3d = self._combine_heads(d_v_split, batch_size)

        # Input gradients
        d_input_q = xp.dot(d_q_3d, self.W_q.T)
        d_input_k = xp.dot(d_k_3d, self.W_k.T)
        d_input_v = xp.dot(d_v_3d, self.W_v.T)

        # Weight gradients
        self.dW_q = xp.dot(self.input_q.transpose(0, 2, 1), d_q_3d).mean(axis=0)
        self.dW_k = xp.dot(self.input_k.transpose(0, 2, 1), d_k_3d).mean(axis=0)
        self.dW_v = xp.dot(self.input_v.transpose(0, 2, 1), d_v_3d).mean(axis=0)
        self.db_q = d_q_3d.mean(axis=(0, 1))
        self.db_k = d_k_3d.mean(axis=(0, 1))
        self.db_v = d_v_3d.mean(axis=(0, 1))

        is_self_attention = (self.input_q is self.input_k) and (self.input_k is self.input_v)
        if is_self_attention:
            return d_input_q + d_input_k + d_input_v
        else:
            return d_input_q, d_input_k, d_input_v

    def get_trainable_params(self) -> Dict:
        return {'W_q': self.W_q, 'W_k': self.W_k, 'W_v': self.W_v, 'W_o': self.W_o,
                'b_q': self.b_q, 'b_k': self.b_k, 'b_v': self.b_v, 'b_o': self.b_o}

    def get_gradients(self) -> Dict:
        return {'W_q': self.dW_q, 'W_k': self.dW_k, 'W_v': self.dW_v, 'W_o': self.dW_o,
                'b_q': self.db_q, 'b_k': self.db_k, 'b_v': self.db_v, 'b_o': self.db_o}

    def update_params(self, params: Dict):
        for key, param in params.items():
            if hasattr(self, key):
                setattr(self, key, self.device.to_device(param))


CrossAttention = MultiHeadAttention