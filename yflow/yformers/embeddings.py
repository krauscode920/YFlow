# yflow/yformers/embeddings.py
"""
Fixed embedding implementations for transformer models.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from ..core.layer import Layer
from ..core.device import Device


class TokenEmbedding(Layer):
    """Token embedding layer that maps token indices to dense vectors."""

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 padding_idx: Optional[int] = None,
                 scale_by_dim: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.scale_by_dim = scale_by_dim

        # Initialize embedding weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize embedding weights using normal distribution"""
        xp = self.device.xp
        std = 0.02

        # Initialize embedding matrix
        self.weight = self.device.to_device(
            xp.random.normal(0, std, (self.vocab_size, self.embed_dim))
        )

        # Initialize padding embeddings to zero if specified
        if self.padding_idx is not None:
            self.weight[self.padding_idx] = 0

    def forward(self, x):
        """Forward pass for token embedding."""
        # Move input to correct device
        x = self.device.to_device(x)
        xp = self.device.xp

        # Store input for backward pass
        self.input = x

        # Convert token indices to embeddings through lookup
        if x.ndim == 1:
            embedded = self.weight[x]
        else:
            # For batch inputs
            batch_size, seq_len = x.shape
            embedded = xp.zeros((batch_size, seq_len, self.embed_dim), dtype=self.weight.dtype)

            # Manual embedding lookup (vectorized across batch)
            for i in range(batch_size):
                embedded[i] = self.weight[x[i]]

        # Scale by dimension if requested
        if self.scale_by_dim:
            embedded = embedded * np.sqrt(self.embed_dim)

        return embedded

    def backward(self, output_gradient):
        """Backward pass for token embedding."""
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)
        xp = self.device.xp

        # Initialize weight gradients
        self.dweight = xp.zeros_like(self.weight)

        # Accumulate gradients for each token index
        if self.input.ndim == 1:
            # For 1D inputs
            for i, idx in enumerate(self.input):
                if self.padding_idx is not None and idx == self.padding_idx:
                    continue
                self.dweight[idx] += output_gradient[i]
        else:
            # For batch inputs
            batch_size, seq_len = self.input.shape
            for b in range(batch_size):
                for s in range(seq_len):
                    idx = self.input[b, s]
                    if self.padding_idx is not None and idx == self.padding_idx:
                        continue
                    self.dweight[idx] += output_gradient[b, s]

        return None

    def get_trainable_params(self) -> Dict:
        return {'weight': self.weight}

    def get_gradients(self) -> Dict:
        return {'weight': self.dweight}

    def update_params(self, params: Dict):
        if 'weight' in params:
            self.weight = self.device.to_device(params['weight'])
            if self.padding_idx is not None:
                self.weight[self.padding_idx] = 0


class PositionalEncoding(Layer):
    """Fixed positional encodings using sine and cosine functions."""

    def __init__(self,
                 embed_dim: int,
                 max_seq_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        self.training = True

        # Pre-compute positional encodings
        self._create_encodings()

    def _create_encodings(self):
        """Create sinusoidal positional encodings."""
        xp = self.device.xp

        # Initialize encoding matrix
        pe = xp.zeros((self.max_seq_len, self.embed_dim))

        # Compute position and dimension indices
        position = xp.arange(0, self.max_seq_len)[:, None]
        div_term = xp.exp(xp.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))

        # Apply sine to even indices
        pe[:, 0::2] = xp.sin(position * div_term)

        # Apply cosine to odd indices
        if self.embed_dim % 2 == 1:
            pe[:, 1::2] = xp.cos(position * div_term)[:, :self.embed_dim // 2]
        else:
            pe[:, 1::2] = xp.cos(position * div_term)

        # Store encodings
        self.pe = self.device.to_device(pe)

    def forward(self, x):
        """Add positional encodings to input embeddings."""
        # Move input to correct device
        x = self.device.to_device(x)
        seq_len = x.shape[1]

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        # Store input for backward pass
        self.input = x

        # Add positional encodings
        x = x + self.pe[:seq_len]

        # Apply dropout if in training mode
        if self.training and self.dropout_rate > 0:
            x = self._apply_dropout(x)

        return x

    def _apply_dropout(self, x):
        """Apply dropout during training"""
        xp = self.device.xp
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            dropout_mask = xp.random.binomial(1, keep_prob, x.shape) / keep_prob
            return x * dropout_mask
        return x

    def backward(self, output_gradient):
        """Backward pass for positional encoding."""
        output_gradient = self.device.to_device(output_gradient)
        return output_gradient

    def get_trainable_params(self) -> Dict:
        return {}

    def get_gradients(self) -> Dict:
        return {}

    def update_params(self, params: Dict):
        pass


class LearnedPositionalEmbedding(Layer):
    """Learnable positional embeddings."""

    def __init__(self,
                 max_seq_len: int,
                 embed_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.dropout_rate = dropout
        self.training = True

        # Initialize embedding weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize position embedding weights"""
        xp = self.device.xp
        std = 0.02

        self.weight = self.device.to_device(
            xp.random.normal(0, std, (self.max_seq_len, self.embed_dim))
        )

    def forward(self, x, start_pos=0):
        """Add learnable positional embeddings to input embeddings."""
        # Move input to correct device
        x = self.device.to_device(x)
        batch_size, seq_len, _ = x.shape

        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {start_pos + seq_len} exceeds maximum {self.max_seq_len}"
            )

        # Store input for backward pass
        self.input = x
        self.start_pos = start_pos
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Add positional embeddings
        positions = self.weight[start_pos:start_pos + seq_len]
        x = x + positions

        # Apply dropout if in training mode
        if self.training and self.dropout_rate > 0:
            x = self._apply_dropout(x)

        return x

    def _apply_dropout(self, x):
        """Apply dropout during training"""
        xp = self.device.xp
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            dropout_mask = xp.random.binomial(1, keep_prob, x.shape) / keep_prob
            self.dropout_mask = dropout_mask
            return x * dropout_mask
        return x

    def backward(self, output_gradient):
        """Backward pass for learnable positional embeddings."""
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)
        xp = self.device.xp

        # Initialize weight gradients
        self.dweight = xp.zeros_like(self.weight)

        # Apply dropout mask if used in forward pass
        if self.training and self.dropout_rate > 0 and hasattr(self, 'dropout_mask'):
            output_gradient = output_gradient * self.dropout_mask

        # FIXED: Accumulate gradients for positions with correct shape handling
        for pos in range(self.seq_len):
            pos_grad = output_gradient[:, pos, :]  # Shape: (batch_size, embed_dim)
            # Sum over batch dimension, don't use mean which can cause shape issues
            self.dweight[self.start_pos + pos] += xp.sum(pos_grad, axis=0)  # Sum over batch

        return output_gradient

    def get_trainable_params(self) -> Dict:
        return {'weight': self.weight}

    def get_gradients(self) -> Dict:
        return {'weight': self.dweight}

    def update_params(self, params: Dict):
        if 'weight' in params:
            self.weight = self.device.to_device(params['weight'])


class PositionalEmbedding(Layer):
    """Combined token and positional embeddings with optional dropout."""

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 max_seq_len: int = 5000,
                 dropout: float = 0.1,
                 padding_idx: Optional[int] = None,
                 learned_pos: bool = False,
                 scale_embed: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout
        self.scale_embed = scale_embed
        self.training = True

        # Create token embeddings
        self.token_embed = TokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            padding_idx=padding_idx,
            scale_by_dim=scale_embed
        )

        # Create positional embeddings
        if learned_pos:
            self.pos_embed = LearnedPositionalEmbedding(
                max_seq_len=max_seq_len,
                embed_dim=embed_dim,
                dropout=0.0
            )
        else:
            self.pos_embed = PositionalEncoding(
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
                dropout=0.0
            )

    def to(self, device_type: str) -> 'PositionalEmbedding':
        """Move layer to specified device"""
        super().to(device_type)
        self.token_embed.to(device_type)
        self.pos_embed.to(device_type)
        return self

    def forward(self, x):
        """Embed tokens and add positional information."""
        # Move input to correct device
        x = self.device.to_device(x)
        self.input = x

        # Get token embeddings
        token_embeddings = self.token_embed.forward(x)

        # Add positional encodings
        embeddings = self.pos_embed.forward(token_embeddings)

        # Apply dropout to final embeddings
        if self.training and self.dropout_rate > 0:
            embeddings = self._apply_dropout(embeddings)

        return embeddings

    def _apply_dropout(self, x):
        """Apply dropout during training"""
        xp = self.device.xp
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            dropout_mask = xp.random.binomial(1, keep_prob, x.shape) / keep_prob
            self.dropout_mask = dropout_mask
            return x * dropout_mask
        return x

    def backward(self, output_gradient):
        """Backward pass for combined embeddings."""
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        # Apply dropout mask if used in forward pass
        if self.training and self.dropout_rate > 0 and hasattr(self, 'dropout_mask'):
            output_gradient = output_gradient * self.dropout_mask

        # Backpropagate through positional embeddings
        pos_grad = self.pos_embed.backward(output_gradient)

        # Backpropagate through token embeddings
        _ = self.token_embed.backward(pos_grad)

        return None

    def get_trainable_params(self) -> Dict:
        """Get trainable parameters from sub-layers"""
        params = {}

        # Get token embedding params with prefix
        token_params = self.token_embed.get_trainable_params()
        for k, v in token_params.items():
            params[f'token_{k}'] = v

        # Get positional embedding params with prefix
        pos_params = self.pos_embed.get_trainable_params()
        for k, v in pos_params.items():
            params[f'pos_{k}'] = v

        return params

    def get_gradients(self) -> Dict:
        """Get parameter gradients from sub-layers"""
        grads = {}

        # Get token embedding gradients with prefix
        token_grads = self.token_embed.get_gradients()
        for k, v in token_grads.items():
            grads[f'token_{k}'] = v

        # Get positional embedding gradients with prefix
        pos_grads = self.pos_embed.get_gradients()
        for k, v in pos_grads.items():
            grads[f'pos_{k}'] = v

        return grads

    def update_params(self, params: Dict):
        """Update parameters in sub-layers"""
        # Filter params for each sub-layer
        token_params = {}
        pos_params = {}

        for key, value in params.items():
            if key.startswith('token_'):
                token_params[key[6:]] = value
            elif key.startswith('pos_'):
                pos_params[key[4:]] = value

        # Update sub-layers
        if token_params:
            self.token_embed.update_params(token_params)
        if pos_params:
            self.pos_embed.update_params(pos_params)