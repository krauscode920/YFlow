# yflow/yformers/embeddings.py
"""
Embedding implementations for transformer models.

This module implements various embedding types used in transformer
architectures, including token embeddings, positional encodings, and combinations of both.
All implementations leverage YFlow's device abstraction for seamless CPU/GPU support.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from ..core.layer import Layer
from ..core.device import Device


class TokenEmbedding(Layer):
    """
    Token embedding layer that maps token indices to dense vectors.

    Args:
        vocab_size (int): Size of the vocabulary
        embed_dim (int): Dimension of the embeddings
        padding_idx (Optional[int]): If specified, the entries at padding_idx
            do not contribute to the gradient; default: None
        scale_by_dim (bool): Whether to scale embeddings by sqrt(embed_dim); default: False
    """

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
        std = 0.02  # Standard deviation commonly used for transformer embeddings

        # Initialize embedding matrix
        self.weight = self.device.to_device(
            xp.random.normal(0, std, (self.vocab_size, self.embed_dim))
        )

        # Initialize padding embeddings to zero if specified
        if self.padding_idx is not None:
            self.weight[self.padding_idx] = 0

    def forward(self, x):
        """
        Forward pass for token embedding.

        Args:
            x: Input tensor of token indices of shape (batch_size, seq_len)

        Returns:
            Embedded representation of shape (batch_size, seq_len, embed_dim)
        """
        # Move input to correct device
        x = self.device.to_device(x)
        xp = self.device.xp

        # Store input for backward pass
        self.input = x

        # Convert token indices to embeddings through lookup
        # Handle both 1D and 2D inputs
        if x.ndim == 1:
            embedded = self.weight[x]  # (seq_len, embed_dim)
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
        """
        Backward pass for token embedding.

        Args:
            output_gradient: Gradient from next layer

        Returns:
            Input gradients (None for index-based inputs)
        """
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
                    continue  # Skip padding tokens
                self.dweight[idx] += output_gradient[i]
        else:
            # For batch inputs
            batch_size, seq_len = self.input.shape
            for b in range(batch_size):
                for s in range(seq_len):
                    idx = self.input[b, s]
                    if self.padding_idx is not None and idx == self.padding_idx:
                        continue  # Skip padding tokens
                    self.dweight[idx] += output_gradient[b, s]

        # No gradient with respect to indices
        return None

    def get_trainable_params(self) -> Dict:
        """Get trainable parameters"""
        return {'weight': self.weight}

    def get_gradients(self) -> Dict:
        """Get parameter gradients"""
        return {'weight': self.dweight}

    def update_params(self, params: Dict):
        """Update layer parameters"""
        if 'weight' in params:
            self.weight = self.device.to_device(params['weight'])
            # Re-zero padding embeddings if specified
            if self.padding_idx is not None:
                self.weight[self.padding_idx] = 0


class PositionalEncoding(Layer):
    """
    Fixed positional encodings using sine and cosine functions as in the
    "Attention Is All You Need" paper.

    Args:
        embed_dim (int): Dimension of the embeddings
        max_seq_len (int): Maximum sequence length to pre-compute
        dropout (float): Dropout rate applied to the positional encodings
    """

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
        """
        Create sinusoidal positional encodings.
        Formula: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
                 PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        """
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
            # Handle odd embedding dimensions
            pe[:, 1::2] = xp.cos(position * div_term)[:, :self.embed_dim // 2]
        else:
            pe[:, 1::2] = xp.cos(position * div_term)

        # Store encodings
        self.register_buffer('pe', pe)

    def register_buffer(self, name, tensor):
        """
        Register a buffer (parameter that is not updated during training)
        """
        setattr(self, name, self.device.to_device(tensor))

    def forward(self, x):
        """
        Add positional encodings to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Embeddings with positional information added
        """
        # Move input to correct device
        x = self.device.to_device(x)

        # Get sequence length from input
        seq_len = x.shape[1]

        # Ensure sequence length is not greater than maximum
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
        """
        Backward pass for positional encoding.

        Args:
            output_gradient: Gradient from next layer

        Returns:
            Gradient with respect to input embeddings
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        # Positional encodings are fixed, so just pass through gradients
        # (applying dropout mask if it was used in forward pass)
        return output_gradient

    def get_trainable_params(self) -> Dict:
        """No trainable parameters for fixed encodings"""
        return {}

    def get_gradients(self) -> Dict:
        """No gradients for fixed encodings"""
        return {}

    def update_params(self, params: Dict):
        """No parameters to update for fixed encodings"""
        pass


class LearnedPositionalEmbedding(Layer):
    """
    Learnable positional embeddings as an alternative to fixed sinusoidal encodings.

    Args:
        max_seq_len (int): Maximum sequence length to learn embeddings for
        embed_dim (int): Dimension of the embeddings
        dropout (float): Dropout rate applied to the positional embeddings
    """

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
        std = 0.02  # Standard deviation commonly used for transformer embeddings

        # Initialize position embeddings
        self.weight = self.device.to_device(
            xp.random.normal(0, std, (self.max_seq_len, self.embed_dim))
        )

    def forward(self, x, start_pos=0):
        """
        Add learnable positional embeddings to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            start_pos: Starting position for the embeddings, useful for
                       incremental decoding; default: 0

        Returns:
            Embeddings with positional information added
        """
        # Move input to correct device
        x = self.device.to_device(x)

        # Get sequence length from input
        batch_size, seq_len, _ = x.shape

        # Ensure sequence length is not greater than maximum
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {start_pos + seq_len} exceeds maximum {self.max_seq_len}"
            )

        # Store input for backward pass
        self.input = x
        self.start_pos = start_pos
        self.seq_len = seq_len

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
            self.dropout_mask = dropout_mask  # Store for backward pass
            return x * dropout_mask
        return x

    def backward(self, output_gradient):
        """
        Backward pass for learnable positional embeddings.

        Args:
            output_gradient: Gradient from next layer

        Returns:
            Gradient with respect to input embeddings
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)
        xp = self.device.xp

        # Initialize weight gradients
        self.dweight = xp.zeros_like(self.weight)

        # Apply dropout mask if used in forward pass
        if self.training and self.dropout_rate > 0 and hasattr(self, 'dropout_mask'):
            output_gradient = output_gradient * self.dropout_mask

        # Accumulate gradients for positions
        batch_size = self.input.shape[0]
        for pos in range(self.seq_len):
            # FIX: Ensure proper shape matching for gradient accumulation
            pos_grad = output_gradient[:, pos, :]  # (batch_size, embed_dim)
            self.dweight[self.start_pos + pos] = xp.mean(pos_grad, axis=0)  # Average over batch

        # Gradient for input is the same as output gradient
        return output_gradient

    def get_trainable_params(self) -> Dict:
        """Get trainable parameters"""
        return {'weight': self.weight}

    def get_gradients(self) -> Dict:
        """Get parameter gradients"""
        return {'weight': self.dweight}

    def update_params(self, params: Dict):
        """Update layer parameters"""
        if 'weight' in params:
            self.weight = self.device.to_device(params['weight'])


class PositionalEmbedding(Layer):
    """
    Combined token and positional embeddings with optional dropout.

    Args:
        vocab_size (int): Size of the vocabulary
        embed_dim (int): Dimension of the embeddings
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout rate applied to the final embeddings
        padding_idx (Optional[int]): If specified, entries at padding_idx do not
                                    contribute to the gradient
        learned_pos (bool): If True, use learnable positional embeddings;
                           otherwise use fixed sinusoidal encodings
        scale_embed (bool): Whether to scale embeddings by sqrt(embed_dim)
    """

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
                dropout=0.0  # We'll apply dropout after combining
            )
        else:
            self.pos_embed = PositionalEncoding(
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
                dropout=0.0  # We'll apply dropout after combining
            )

    def to(self, device_type: str) -> 'PositionalEmbedding':
        """Move layer to specified device"""
        super().to(device_type)
        self.token_embed.to(device_type)
        self.pos_embed.to(device_type)
        return self

    def forward(self, x):
        """
        Embed tokens and add positional information.

        Args:
            x: Input tensor of token indices of shape (batch_size, seq_len)

        Returns:
            Combined embeddings of shape (batch_size, seq_len, embed_dim)
        """
        # Move input to correct device
        x = self.device.to_device(x)

        # Store input for backward pass
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
            self.dropout_mask = dropout_mask  # Store for backward pass
            return x * dropout_mask
        return x

    def backward(self, output_gradient):
        """
        Backward pass for combined embeddings.

        Args:
            output_gradient: Gradient from next layer

        Returns:
            None (since input is token indices)
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        # Apply dropout mask if used in forward pass
        if self.training and self.dropout_rate > 0 and hasattr(self, 'dropout_mask'):
            output_gradient = output_gradient * self.dropout_mask

        # Backpropagate through positional embeddings
        pos_grad = self.pos_embed.backward(output_gradient)

        # Backpropagate through token embeddings
        _ = self.token_embed.backward(pos_grad)

        # No gradient with respect to token indices
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
                token_params[key[6:]] = value  # Remove 'token_' prefix
            elif key.startswith('pos_'):
                pos_params[key[4:]] = value  # Remove 'pos_' prefix

        # Update sub-layers
        if token_params:
            self.token_embed.update_params(token_params)
        if pos_params:
            self.pos_embed.update_params(pos_params)