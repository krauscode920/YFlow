# yflow/yformers/model.py
"""
Complete transformer model implementations.

This module implements various transformer architectures by combining
components such as encoders, decoders, and embeddings into full models.
All implementations leverage YFlow's device abstraction for seamless CPU/GPU support.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, List, Any
from ..core.model import Model
from ..core.layer import Layer
from ..core.device import Device
from .attention import MultiHeadAttention
from .embeddings import (
    TokenEmbedding,
    PositionalEmbedding,
    PositionalEncoding
)
from .encoder import EncoderStack
from .decoder import DecoderStack
from ..layers.normalization import LayerNorm
from ..layers.dense import Dense


# Add this class at the beginning of your yflow/yformers/model.py file
# Place it right after the imports and before the TransformerModel class

class TransformerBase(Model):
    """
    Base class for all transformer models.

    Provides common functionality and interfaces for transformer implementations.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize transformer base with optional random seed"""
        super().__init__(seed=seed)
        self.training = True

    def create_padding_mask(self, x):
        """
        Create padding mask for encoder attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Padding mask of shape (batch_size, 1, 1, seq_len)
        """
        xp = self.device.xp
        # Mask where input tokens are padding (0)
        mask = (x != 0).astype(xp.float32)
        # Add broadcast dimensions for attention heads
        return mask[:, None, None, :]

    def create_look_ahead_mask(self, seq_len):
        """
        Create causal mask to prevent attention to future tokens.

        Args:
            seq_len: Length of sequence

        Returns:
            Causal mask of shape (seq_len, seq_len)
        """
        xp = self.device.xp
        # Create lower triangular matrix
        mask = xp.tril(xp.ones((seq_len, seq_len)))
        return mask

    def create_combined_mask(self, x):
        """
        Create combined padding and causal mask for decoder self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Combined mask of shape (batch_size, 1, seq_len, seq_len)
        """
        xp = self.device.xp
        seq_len = x.shape[1]

        # Create padding mask (batch_size, 1, 1, seq_len)
        padding_mask = self.create_padding_mask(x)

        # Create causal mask (1, 1, seq_len, seq_len)
        look_ahead_mask = self.create_look_ahead_mask(seq_len)[None, None, :, :]

        # Combine masks
        combined_mask = xp.minimum(padding_mask[:, :, :, None], look_ahead_mask)
        return combined_mask

    def get_config(self) -> Dict:
        """Get model configuration"""
        return {
            'model_type': self.__class__.__name__,
            'device': self.device.device_type
        }

# Add this to your yflow/yformers/model.py file
# Replace the duplicate TransformerModel class with this fixed version

class TransformerModel(TransformerBase):
    """
    Complete encoder-decoder transformer as described in "Attention Is All You Need".
    """

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1,
                 max_src_len: int = 5000,
                 max_tgt_len: int = 5000,
                 share_embeddings: bool = False,
                 pre_norm: bool = True,
                 activation: str = 'relu',
                 seed: Optional[int] = None):
        super().__init__(seed=seed)

        # Store model configuration
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.share_embeddings = share_embeddings
        self.pre_norm = pre_norm

        # Check that embedding dimension is compatible with number of heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        # Initialize components
        # Encoder embedding
        self.encoder_embedding = PositionalEmbedding(
            vocab_size=src_vocab_size,
            embed_dim=d_model,
            max_seq_len=max_src_len,
            dropout=dropout,
            padding_idx=0,
            learned_pos=False,
            scale_embed=True
        )

        # Decoder embedding
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.decoder_embedding = PositionalEmbedding(
                vocab_size=tgt_vocab_size,
                embed_dim=d_model,
                max_seq_len=max_tgt_len,
                dropout=dropout,
                padding_idx=0,
                learned_pos=False,
                scale_embed=True
            )
            # Share token embeddings' weights
            self.decoder_embedding.token_embed.weight = self.encoder_embedding.token_embed.weight
        else:
            self.decoder_embedding = PositionalEmbedding(
                vocab_size=tgt_vocab_size,
                embed_dim=d_model,
                max_seq_len=max_tgt_len,
                dropout=dropout,
                padding_idx=0,
                learned_pos=False,
                scale_embed=True
            )

        # Encoder
        self.encoder = EncoderStack(
            embed_dim=d_model,
            num_heads=num_heads,
            ff_dim=d_ff,
            num_layers=num_encoder_layers,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=activation
        )

        # Decoder
        self.decoder = DecoderStack(
            embed_dim=d_model,
            num_heads=num_heads,
            ff_dim=d_ff,
            num_layers=num_decoder_layers,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=activation
        )

        # Output projection
        self.output_proj = Dense(
            input_size=d_model,
            output_size=tgt_vocab_size
        )

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, cross_mask=None):
        """Forward pass through the complete transformer model."""
        # Move inputs to correct device
        src = self.device.to_device(src)
        if tgt is not None:
            tgt = self.device.to_device(tgt)

        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        else:
            src_mask = self.device.to_device(src_mask)

        # For inference, tgt might be None
        if tgt is None:
            xp = self.device.xp
            tgt = xp.ones((src.shape[0], 1), dtype=xp.int32)

        if tgt_mask is None:
            tgt_mask = self.create_combined_mask(tgt)
        else:
            tgt_mask = self.device.to_device(tgt_mask)

        if cross_mask is None:
            cross_mask = self.create_padding_mask(src)
        else:
            cross_mask = self.device.to_device(cross_mask)

        # Encoder path
        enc_input = self.encoder_embedding.forward(src)
        enc_output = self.encoder.forward(enc_input, mask=src_mask)

        # Decoder path
        dec_input = self.decoder_embedding.forward(tgt)
        dec_output = self.decoder.forward(
            dec_input,
            enc_output,
            self_attn_mask=tgt_mask,
            cross_attn_mask=cross_mask
        )

        # Output projection
        logits = self.output_proj.forward(dec_output)

        # Cache inputs and outputs for backward pass
        self.cache = {
            'src': src,
            'tgt': tgt,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
            'cross_mask': cross_mask,
            'enc_input': enc_input,
            'enc_output': enc_output,
            'dec_input': dec_input,
            'dec_output': dec_output
        }

        return logits

    def backward(self, output_gradient):
        """Backward pass through the complete transformer model."""
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        # Backpropagate through output projection
        dec_output_grad = self.output_proj.backward(output_gradient)

        # Backpropagate through decoder
        dec_input_grad, enc_output_grad = self.decoder.backward(dec_output_grad)

        # Backpropagate through decoder embedding
        _ = self.decoder_embedding.backward(dec_input_grad)

        # Backpropagate through encoder
        enc_input_grad = self.encoder.backward(enc_output_grad)

        # Backpropagate through encoder embedding
        _ = self.encoder_embedding.backward(enc_input_grad)

        return None, None

    # Fix for yflow/yformers/model.py
    # In the TransformerModel class, replace the generate method:

    def generate(self, src, max_len=None, temperature=1.0, top_k=0, top_p=0.0):
        """Generate a target sequence from source sequence."""
        if max_len is None:
            max_len = self.max_tgt_len

        # Move inputs to correct device
        src = self.device.to_device(src)
        xp = self.device.xp

        # Encoder only needs to run once
        enc_input = self.encoder_embedding.forward(src)
        enc_output = self.encoder.forward(enc_input, mask=self.create_padding_mask(src))

        # Initialize with BOS token (assuming token ID 1)
        batch_size = src.shape[0]
        output = xp.ones((batch_size, 1), dtype=xp.int32)

        # Generate tokens auto-regressively
        for i in range(max_len - 1):
            # Predict next token
            logits = self.forward(
                src,
                output,
                src_mask=self.create_padding_mask(src),
                tgt_mask=self.create_combined_mask(output),
                cross_mask=self.create_padding_mask(src)
            )

            # Get logits for the last position
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Manual softmax implementation
            exp_logits = xp.exp(next_token_logits - xp.max(next_token_logits, axis=-1, keepdims=True))
            probs = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)

            # Simple sampling (most probable token)
            next_token = xp.argmax(probs, axis=-1).reshape(-1, 1)

            # Append to output
            output = xp.concatenate([output, next_token], axis=1)

            # Check if all sequences have hit EOS token (assuming token ID 2)
            if xp.all(next_token == 2):
                break

        return output

    def to(self, device_type: str) -> 'TransformerModel':
        """Move model to specified device"""
        super().to(device_type)
        self.encoder_embedding.to(device_type)
        self.decoder_embedding.to(device_type)
        self.encoder.to(device_type)
        self.decoder.to(device_type)
        self.output_proj.to(device_type)
        return self

    def get_trainable_params(self) -> Dict:
        """Get all trainable parameters"""
        params = {}

        # Get parameters from encoder embedding
        enc_emb_params = self.encoder_embedding.get_trainable_params()
        for key, value in enc_emb_params.items():
            params[f'enc_emb_{key}'] = value

        # Get parameters from decoder embedding (if not shared)
        if not self.share_embeddings or self.src_vocab_size != self.tgt_vocab_size:
            dec_emb_params = self.decoder_embedding.get_trainable_params()
            for key, value in dec_emb_params.items():
                params[f'dec_emb_{key}'] = value

        # Get parameters from encoder
        enc_params = self.encoder.get_trainable_params()
        for key, value in enc_params.items():
            params[f'enc_{key}'] = value

        # Get parameters from decoder
        dec_params = self.decoder.get_trainable_params()
        for key, value in dec_params.items():
            params[f'dec_{key}'] = value

        # Get parameters from output projection
        out_params = self.output_proj.get_trainable_params()
        for key, value in out_params.items():
            params[f'out_{key}'] = value

        return params

    def get_gradients(self) -> Dict:
        """Get all parameter gradients"""
        grads = {}

        # Get gradients from encoder embedding
        enc_emb_grads = self.encoder_embedding.get_gradients()
        for key, value in enc_emb_grads.items():
            grads[f'enc_emb_{key}'] = value

        # Get gradients from decoder embedding (if not shared)
        if not self.share_embeddings or self.src_vocab_size != self.tgt_vocab_size:
            dec_emb_grads = self.decoder_embedding.get_gradients()
            for key, value in dec_emb_grads.items():
                grads[f'dec_emb_{key}'] = value

        # Get gradients from encoder
        enc_grads = self.encoder.get_gradients()
        for key, value in enc_grads.items():
            grads[f'enc_{key}'] = value

        # Get gradients from decoder
        dec_grads = self.decoder.get_gradients()
        for key, value in dec_grads.items():
            grads[f'dec_{key}'] = value

        # Get gradients from output projection
        out_grads = self.output_proj.get_gradients()
        for key, value in out_grads.items():
            grads[f'out_{key}'] = value

        return grads

    def update_params(self, params: Dict):
        """Update all parameters"""
        # Group parameters by component
        enc_emb_params = {}
        dec_emb_params = {}
        enc_params = {}
        dec_params = {}
        out_params = {}

        for key, value in params.items():
            if key.startswith('enc_emb_'):
                enc_emb_params[key[8:]] = value
            elif key.startswith('dec_emb_'):
                dec_emb_params[key[8:]] = value
            elif key.startswith('enc_'):
                enc_params[key[4:]] = value
            elif key.startswith('dec_'):
                dec_params[key[4:]] = value
            elif key.startswith('out_'):
                out_params[key[4:]] = value

        # Update components
        if enc_emb_params:
            self.encoder_embedding.update_params(enc_emb_params)
        if dec_emb_params:
            self.decoder_embedding.update_params(dec_emb_params)
        if enc_params:
            self.encoder.update_params(enc_params)
        if dec_params:
            self.decoder.update_params(dec_params)
        if out_params:
            self.output_proj.update_params(out_params)

class EncoderOnlyModel(TransformerBase):
    """
    Encoder-only transformer model for sequence classification or encoding.

    Similar to BERT-style models, this uses only the encoder stack
    and is well-suited for classification, feature extraction, etc.

    Args:
        vocab_size (int): Size of vocabulary
        d_model (int): Embedding dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward inner dimension
        num_layers (int): Number of encoder blocks
        dropout (float): Dropout rate
        max_seq_len (int): Maximum sequence length
        pre_norm (bool): Whether to use Pre-LN (True) or Post-LN (False) architecture
        activation (str): Activation function for feed-forward networks
        num_classes (int): Number of output classes (0 for no classification head)
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 768,
                 num_heads: int = 12,
                 d_ff: int = 3072,
                 num_layers: int = 12,
                 dropout: float = 0.1,
                 max_seq_len: int = 512,
                 pre_norm: bool = True,
                 activation: str = 'gelu',
                 num_classes: int = 0,
                 seed: Optional[int] = None):
        super().__init__(seed=seed)

        # Store model configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.pre_norm = pre_norm
        self.num_classes = num_classes

        # Check that embedding dimension is compatible with number of heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        # Initialize components
        # Token and position embeddings
        self.embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            embed_dim=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=0,
            learned_pos=True,  # Use learned positional embeddings like BERT
            scale_embed=False
        )

        # Encoder stack
        self.encoder = EncoderStack(
            embed_dim=d_model,
            num_heads=num_heads,
            ff_dim=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=activation
        )

        # Classification head if specified
        self.classifier = None
        if num_classes > 0:
            self.classifier = Dense(
                input_size=d_model,
                output_size=num_classes
            )

    def forward(self, tokens, attention_mask=None):
        """
        Forward pass through the encoder-only model.

        Args:
            tokens: Input tokens of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            If num_classes > 0: Classification logits of shape (batch_size, num_classes)
            Otherwise: Encoded representation of shape (batch_size, seq_len, d_model)
        """
        # Move inputs to correct device
        tokens = self.device.to_device(tokens)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_padding_mask(tokens)
        else:
            attention_mask = self.device.to_device(attention_mask)

        # Embedding
        x = self.embedding.forward(tokens)

        # Encoder
        encoded = self.encoder.forward(x, mask=attention_mask)

        # Classification if specified
        if self.classifier is not None:
            # Use [CLS] token representation (first token)
            cls_output = encoded[:, 0, :]
            logits = self.classifier.forward(cls_output)

            # Cache for backward pass
            self.cache = {
                'tokens': tokens,
                'attention_mask': attention_mask,
                'embedded': x,
                'encoded': encoded,
                'cls_output': cls_output
            }

            return logits

        # Return full encoded representation
        self.cache = {
            'tokens': tokens,
            'attention_mask': attention_mask,
            'embedded': x
        }

        return encoded

    def backward(self, output_gradient):
        """
        Backward pass through the encoder-only model.

        Args:
            output_gradient: Gradient from loss function or next layer

        Returns:
            None (since input is token indices)
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        # Backpropagate through classifier if used
        if self.classifier is not None:
            cls_grad = self.classifier.backward(output_gradient)

            # Expand gradient to full encoded sequence
            xp = self.device.xp
            encoded_grad = xp.zeros_like(self.cache['encoded'])
            encoded_grad[:, 0, :] = cls_grad
        else:
            # Output gradient is already for the full sequence
            encoded_grad = output_gradient

        # Backpropagate through encoder
        embedded_grad = self.encoder.backward(encoded_grad)

        # Backpropagate through embedding
        _ = self.embedding.backward(embedded_grad)

        # No gradient with respect to token indices
        return None

    def to(self, device_type: str) -> 'EncoderOnlyModel':
        """Move model to specified device"""
        super().to(device_type)
        self.embedding.to(device_type)
        self.encoder.to(device_type)
        if self.classifier is not None:
            self.classifier.to(device_type)
        return self

    def get_embedding(self, tokens):
        """
        Get token embeddings without full model forward pass.

        Args:
            tokens: Input tokens of shape (batch_size, seq_len)

        Returns:
            Token embeddings of shape (batch_size, seq_len, d_model)
        """
        tokens = self.device.to_device(tokens)
        return self.embedding.forward(tokens)

    def get_config(self) -> Dict:
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'pre_norm': self.pre_norm,
            'num_classes': self.num_classes
        })
        return config

    def get_trainable_params(self) -> Dict:
        """Get all trainable parameters"""
        params = {}

        # Get parameters from embedding
        emb_params = self.embedding.get_trainable_params()
        for key, value in emb_params.items():
            params[f'emb_{key}'] = value

        # Get parameters from encoder
        enc_params = self.encoder.get_trainable_params()
        for key, value in enc_params.items():
            params[f'enc_{key}'] = value

        # Get parameters from classifier if used
        if self.classifier is not None:
            cls_params = self.classifier.get_trainable_params()
            for key, value in cls_params.items():
                params[f'cls_{key}'] = value

        return params

    def get_gradients(self) -> Dict:
        """Get all parameter gradients"""
        grads = {}

        # Get gradients from embedding
        emb_grads = self.embedding.get_gradients()
        for key, value in emb_grads.items():
            grads[f'emb_{key}'] = value

        # Get gradients from encoder
        enc_grads = self.encoder.get_gradients()
        for key, value in enc_grads.items():
            grads[f'enc_{key}'] = value

        # Get gradients from classifier if used
        if self.classifier is not None:
            cls_grads = self.classifier.get_gradients()
            for key, value in cls_grads.items():
                grads[f'cls_{key}'] = value

        return grads

    def update_params(self, params: Dict):
        """Update all parameters"""
        # Group parameters by component
        emb_params = {}
        enc_params = {}
        cls_params = {}

        for key, value in params.items():
            if key.startswith('emb_'):
                emb_params[key[4:]] = value
            elif key.startswith('enc_'):
                enc_params[key[4:]] = value
            elif key.startswith('cls_'):
                cls_params[key[4:]] = value

        # Update components
        if emb_params:
            self.embedding.update_params(emb_params)
        if enc_params:
            self.encoder.update_params(enc_params)
        if cls_params and self.classifier is not None:
            self.classifier.update_params(cls_params)


class DecoderOnlyModel(TransformerBase):
    """
    Decoder-only transformer model for language generation.

    Similar to GPT-style models, this uses only the decoder stack
    (with causal attention) and is well-suited for text generation tasks.

    Args:
        vocab_size (int): Size of vocabulary
        d_model (int): Embedding dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward inner dimension
        num_layers (int): Number of decoder blocks
        dropout (float): Dropout rate
        max_seq_len (int): Maximum sequence length
        pre_norm (bool): Whether to use Pre-LN (True) or Post-LN (False) architecture
        activation (str): Activation function for feed-forward networks
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 768,
                 num_heads: int = 12,
                 d_ff: int = 3072,
                 num_layers: int = 12,
                 dropout: float = 0.1,
                 max_seq_len: int = 1024,
                 pre_norm: bool = True,
                 activation: str = 'gelu',
                 seed: Optional[int] = None):
        super().__init__(seed=seed)

        # Store model configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.pre_norm = pre_norm

        # Check that embedding dimension is compatible with number of heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        # Initialize components
        # Token and position embeddings
        self.embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            embed_dim=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=0,
            learned_pos=True,  # Use learned positional embeddings like GPT
            scale_embed=False
        )

        # Create custom decoder blocks that only have self-attention (no cross-attention)
        # We'll reuse EncoderStack but ensure masks are properly handled
        self.decoder = EncoderStack(
            embed_dim=d_model,
            num_heads=num_heads,
            ff_dim=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=activation
        )

        # Output projection (tied with input embeddings)
        self.output_proj = Dense(
            input_size=d_model,
            output_size=vocab_size
        )

        # Tie weights with embedding if compatible
        self.tie_weights()

    def tie_weights(self):
        """Tie output projection weights with input embeddings"""
        # Make output projection's weights the same as embedding weights
        self.output_proj.weights = self.embedding.token_embed.weight.T

    def forward(self, tokens, attention_mask=None):
        """
        Forward pass through the decoder-only model.

        Args:
            tokens: Input tokens of shape (batch_size, seq_len)
            attention_mask: Optional custom attention mask

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Move inputs to correct device
        tokens = self.device.to_device(tokens)

        # Create causal attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_combined_mask(tokens)
        else:
            attention_mask = self.device.to_device(attention_mask)

        # Embedding
        x = self.embedding.forward(tokens)

        # Decoder (causal self-attention only)
        decoded = self.decoder.forward(x, mask=attention_mask)

        # Output projection
        logits = self.output_proj.forward(decoded)

        # Cache for backward pass
        self.cache = {
            'tokens': tokens,
            'attention_mask': attention_mask,
            'embedded': x,
            'decoded': decoded
        }

        return logits

    def backward(self, output_gradient):
        """
        Backward pass through the decoder-only model.

        Args:
            output_gradient: Gradient from loss function

        Returns:
            None (since input is token indices)
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        # Backpropagate through output projection
        decoded_grad = self.output_proj.backward(output_gradient)

        # Backpropagate through decoder
        embedded_grad = self.decoder.backward(decoded_grad)

        # Backpropagate through embedding
        _ = self.embedding.backward(embedded_grad)

        # Weight tying gradient accumulation
        if hasattr(self, 'output_proj') and hasattr(self, 'embedding'):
            # Add the transposed output projection gradient to the embedding gradient
            self.embedding.token_embed.dweight += self.output_proj.weights_gradient.T

        # No gradient with respect to token indices
        return None

    # Fix for yflow/yformers/model.py
    # In the DecoderOnlyModel class, replace the generate method:

    def generate(self, prompt_tokens, max_len=None, temperature=1.0, top_k=0, top_p=0.0):
        """Generate text from a prompt."""
        if max_len is None:
            max_len = self.max_seq_len

        # Move inputs to correct device
        prompt_tokens = self.device.to_device(prompt_tokens)
        xp = self.device.xp

        batch_size, prompt_len = prompt_tokens.shape

        # Initialize with prompt
        output = prompt_tokens

        # Generate tokens auto-regressively
        for i in range(max_len - prompt_len):
            # Create appropriate attention mask
            attention_mask = self.create_combined_mask(output)

            # Forward pass through the model
            logits = self.forward(output, attention_mask=attention_mask)

            # Get logits for the last position
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering if specified
            if top_k > 0:
                # Get the top-k values
                sorted_indices = xp.argsort(next_token_logits, axis=-1)[:, ::-1]
                top_k_indices = sorted_indices[:, :top_k]

                # Create mask for top-k values
                mask = xp.zeros_like(next_token_logits)
                for b in range(batch_size):
                    mask[b, top_k_indices[b]] = 1

                # Apply mask
                next_token_logits = xp.where(mask, next_token_logits, -float('inf'))

            # Manual softmax implementation (numerically stable)
            exp_logits = xp.exp(next_token_logits - xp.max(next_token_logits, axis=-1, keepdims=True))
            probs = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)

            # Sample next token (using argmax for simplicity)
            next_token = xp.argmax(probs, axis=-1).reshape(-1, 1)

            # Append to output
            output = xp.concatenate([output, next_token], axis=1)

            # Check for EOS token (assuming token ID 2)
            if xp.all(next_token == 2):
                break

        return output

    def to(self, device_type: str) -> 'DecoderOnlyModel':
        """Move model to specified device"""
        super().to(device_type)
        self.embedding.to(device_type)
        self.decoder.to(device_type)
        self.output_proj.to(device_type)
        return self

    def get_config(self) -> Dict:
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'pre_norm': self.pre_norm
        })
        return config

    def get_trainable_params(self) -> Dict:
        """Get all trainable parameters"""
        params = {}

        # Get parameters from embedding
        emb_params = self.embedding.get_trainable_params()
        for key, value in emb_params.items():
            params[f'emb_{key}'] = value

        # Get parameters from decoder
        dec_params = self.decoder.get_trainable_params()
        for key, value in dec_params.items():
            params[f'dec_{key}'] = value

        # Do not include output projection since weights are tied

        return params

    def get_gradients(self) -> Dict:
        """Get all parameter gradients"""
        grads = {}

        # Get gradients from embedding
        emb_grads = self.embedding.get_gradients()
        for key, value in emb_grads.items():
            grads[f'emb_{key}'] = value

        # Get gradients from decoder
        dec_grads = self.decoder.get_gradients()
        for key, value in dec_grads.items():
            grads[f'dec_{key}'] = value

        # Do not include output projection since weights are tied

        return grads

    def update_params(self, params: Dict):
        """Update all parameters"""
        # Group parameters by component
        emb_params = {}
        dec_params = {}

        for key, value in params.items():
            if key.startswith('emb_'):
                emb_params[key[4:]] = value
            elif key.startswith('dec_'):
                dec_params[key[4:]] = value

        # Update components
        if emb_params:
            self.embedding.update_params(emb_params)
        if dec_params:
            self.decoder.update_params(dec_params)

        # Re-tie weights after update
        self.tie_weights()