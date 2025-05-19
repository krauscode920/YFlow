# yflow/yformers/decoder.py
"""
Decoder components for transformer models.

This module implements the decoder blocks and stacks used in transformer architectures.
All implementations leverage YFlow's device abstraction for seamless CPU/GPU support.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, List
from ..core.layer import Layer
from ..core.device import Device
from .attention import MultiHeadAttention
from ..layers.normalization import LayerNorm
from .encoder import FeedForward


class DecoderBlock(Layer):
    """
    Transformer decoder block.

    Each block consists of:
    1. Masked multi-head self-attention with residual connection and layer norm
    2. Multi-head cross-attention with residual connection and layer norm
    3. Feed-forward network with residual connection and layer norm

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward inner dimension
        dropout (float): Dropout rate
        pre_norm (bool): If True, apply layer norm before attention and feed-forward
                        (Pre-LN) rather than after (Post-LN)
        activation (str): Activation function for feed-forward network
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 pre_norm: bool = True,
                 activation: str = 'relu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.pre_norm = pre_norm
        self.training = True

        # Initialize components
        # Self-attention (masked)
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Cross-attention
        self.cross_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Layer normalizations
        self.ln1 = LayerNorm(embed_dim)  # For self-attention
        self.ln2 = LayerNorm(embed_dim)  # For cross-attention
        self.ln3 = LayerNorm(embed_dim)  # For feed-forward

        # Feed-forward network
        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation
        )

    def to(self, device_type: str) -> 'DecoderBlock':
        """Move layer to specified device"""
        super().to(device_type)
        self.self_attn.to(device_type)
        self.cross_attn.to(device_type)
        self.ln1.to(device_type)
        self.ln2.to(device_type)
        self.ln3.to(device_type)
        self.feed_forward.to(device_type)
        return self

    def _apply_dropout(self, x):
        """Apply dropout during training"""
        xp = self.device.xp
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            dropout_mask = xp.random.binomial(1, keep_prob, x.shape) / keep_prob
            return x * dropout_mask
        return x

    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        Forward pass for decoder block.

        Args:
            x: Input tensor of shape (batch_size, tgt_seq_len, embed_dim)
            encoder_output: Output from encoder of shape (batch_size, src_seq_len, embed_dim)
            self_attn_mask: Mask for self-attention to prevent looking at future tokens
                           Shape: (batch_size, tgt_seq_len, tgt_seq_len)
            cross_attn_mask: Mask for cross-attention to handle padding in encoder output
                           Shape: (batch_size, tgt_seq_len, src_seq_len)

        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, embed_dim)
        """
        # Move inputs to correct device
        x = self.device.to_device(x)
        encoder_output = self.device.to_device(encoder_output)
        if self_attn_mask is not None:
            self_attn_mask = self.device.to_device(self_attn_mask)
        if cross_attn_mask is not None:
            cross_attn_mask = self.device.to_device(cross_attn_mask)

        # Store inputs for backward pass
        self.input = x
        self.encoder_output = encoder_output

        if self.pre_norm:
            # Pre-LN architecture
            # 1. Self-attention with residual connection
            norm1 = self.ln1.forward(x)
            self_attn_output = self.self_attn.forward(norm1, mask=self_attn_mask)
            self_attn_output = self._apply_dropout(self_attn_output)
            self.self_attn_dropout_mask = getattr(self, 'dropout_mask', None)
            x1 = x + self_attn_output

            # 2. Cross-attention with residual connection
            norm2 = self.ln2.forward(x1)
            cross_attn_output = self.cross_attn.forward(norm2, encoder_output, encoder_output, mask=cross_attn_mask)
            cross_attn_output = self._apply_dropout(cross_attn_output)
            self.cross_attn_dropout_mask = getattr(self, 'dropout_mask', None)
            x2 = x1 + cross_attn_output

            # 3. Feed-forward with residual connection
            norm3 = self.ln3.forward(x2)
            ff_output = self.feed_forward.forward(norm3)
            ff_output = self._apply_dropout(ff_output)
            self.ff_dropout_mask = getattr(self, 'dropout_mask', None)
            output = x2 + ff_output
        else:
            # Post-LN architecture
            # 1. Self-attention with residual connection
            self_attn_output = self.self_attn.forward(x, mask=self_attn_mask)
            self_attn_output = self._apply_dropout(self_attn_output)
            self.self_attn_dropout_mask = getattr(self, 'dropout_mask', None)
            x1 = self.ln1.forward(x + self_attn_output)

            # 2. Cross-attention with residual connection
            cross_attn_output = self.cross_attn.forward(x1, encoder_output, encoder_output, mask=cross_attn_mask)
            cross_attn_output = self._apply_dropout(cross_attn_output)
            self.cross_attn_dropout_mask = getattr(self, 'dropout_mask', None)
            x2 = self.ln2.forward(x1 + cross_attn_output)

            # 3. Feed-forward with residual connection
            ff_output = self.feed_forward.forward(x2)
            ff_output = self._apply_dropout(ff_output)
            self.ff_dropout_mask = getattr(self, 'dropout_mask', None)
            output = self.ln3.forward(x2 + ff_output)

        # Save intermediate outputs for backward pass
        self.self_attn_output = self_attn_output
        self.cross_attn_output = cross_attn_output
        self.ff_output = ff_output
        self.x1 = x1
        self.x2 = x2

        return output

    def backward(self, output_gradient):
        """
        Backward pass for decoder block.

        Args:
            output_gradient: Gradient from next layer

        Returns:
            Tuple of gradients: (input_gradient, encoder_output_gradient)
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        if self.pre_norm:
            # Pre-LN architecture backward pass
            # 1. Gradient through feed-forward residual connection
            ff_grad = output_gradient
            x2_grad = output_gradient

            # Apply feed-forward dropout gradient if used
            if self.training and self.dropout_rate > 0 and self.ff_dropout_mask is not None:
                ff_grad = ff_grad * self.ff_dropout_mask

            # Backpropagate through feed-forward
            norm3_grad = self.feed_forward.backward(ff_grad)

            # Backpropagate through layer norm
            x2_grad2 = self.ln3.backward(norm3_grad)

            # Sum gradients at x2
            x2_grad = x2_grad + x2_grad2

            # 2. Gradient through cross-attention residual connection
            cross_attn_grad = x2_grad
            x1_grad = x2_grad

            # Apply cross-attention dropout gradient if used
            if self.training and self.dropout_rate > 0 and self.cross_attn_dropout_mask is not None:
                cross_attn_grad = cross_attn_grad * self.cross_attn_dropout_mask

            # Backpropagate through cross-attention
            norm2_grad, encoder_output_grad = self.cross_attn.backward(cross_attn_grad)

            # Backpropagate through layer norm
            x1_grad2 = self.ln2.backward(norm2_grad)

            # Sum gradients at x1
            x1_grad = x1_grad + x1_grad2

            # 3. Gradient through self-attention residual connection
            self_attn_grad = x1_grad
            x_grad = x1_grad

            # Apply self-attention dropout gradient if used
            if self.training and self.dropout_rate > 0 and self.self_attn_dropout_mask is not None:
                self_attn_grad = self_attn_grad * self.self_attn_dropout_mask

            # Backpropagate through self-attention
            norm1_grad = self.self_attn.backward(self_attn_grad)

            # Backpropagate through layer norm
            x_grad2 = self.ln1.backward(norm1_grad)

            # Sum gradients at x
            x_grad = x_grad + x_grad2

        else:
            # Post-LN architecture backward pass
            # 1. Backpropagate through final layer norm
            x2_plus_ff_grad = self.ln3.backward(output_gradient)

            # Gradient through feed-forward residual connection
            ff_grad = x2_plus_ff_grad
            x2_grad = x2_plus_ff_grad

            # Apply feed-forward dropout gradient if used
            if self.training and self.dropout_rate > 0 and self.ff_dropout_mask is not None:
                ff_grad = ff_grad * self.ff_dropout_mask

            # Backpropagate through feed-forward
            x2_grad2 = self.feed_forward.backward(ff_grad)

            # Sum gradients at x2
            x2_grad = x2_grad + x2_grad2

            # 2. Backpropagate through second layer norm
            x1_plus_cross_attn_grad = self.ln2.backward(x2_grad)

            # Gradient through cross-attention residual connection
            cross_attn_grad = x1_plus_cross_attn_grad
            x1_grad = x1_plus_cross_attn_grad

            # Apply cross-attention dropout gradient if used
            if self.training and self.dropout_rate > 0 and self.cross_attn_dropout_mask is not None:
                cross_attn_grad = cross_attn_grad * self.cross_attn_dropout_mask

            # Backpropagate through cross-attention
            x1_grad2, encoder_output_grad = self.cross_attn.backward(cross_attn_grad)

            # Sum gradients at x1
            x1_grad = x1_grad + x1_grad2

            # 3. Backpropagate through first layer norm
            x_plus_self_attn_grad = self.ln1.backward(x1_grad)

            # Gradient through self-attention residual connection
            self_attn_grad = x_plus_self_attn_grad
            x_grad = x_plus_self_attn_grad

            # Apply self-attention dropout gradient if used
            if self.training and self.dropout_rate > 0 and self.self_attn_dropout_mask is not None:
                self_attn_grad = self_attn_grad * self.self_attn_dropout_mask

            # Backpropagate through self-attention
            x_grad2 = self.self_attn.backward(self_attn_grad)

            # Sum gradients at x
            x_grad = x_grad + x_grad2

        return x_grad, encoder_output_grad

    def get_trainable_params(self) -> Dict:
        """Get trainable parameters from sub-components"""
        params = {}

        # Get parameters from self-attention
        self_attn_params = self.self_attn.get_trainable_params()
        for key, value in self_attn_params.items():
            params[f'self_attn_{key}'] = value

        # Get parameters from cross-attention
        cross_attn_params = self.cross_attn.get_trainable_params()
        for key, value in cross_attn_params.items():
            params[f'cross_attn_{key}'] = value

        # Get parameters from feed-forward
        ff_params = self.feed_forward.get_trainable_params()
        for key, value in ff_params.items():
            params[f'ff_{key}'] = value

        # Get parameters from layer norms
        ln1_params = self.ln1.get_trainable_params()
        for key, value in ln1_params.items():
            params[f'ln1_{key}'] = value

        ln2_params = self.ln2.get_trainable_params()
        for key, value in ln2_params.items():
            params[f'ln2_{key}'] = value

        ln3_params = self.ln3.get_trainable_params()
        for key, value in ln3_params.items():
            params[f'ln3_{key}'] = value

        return params

    def get_gradients(self) -> Dict:
        """Get parameter gradients from sub-components"""
        grads = {}

        # Get gradients from self-attention
        self_attn_grads = self.self_attn.get_gradients()
        for key, value in self_attn_grads.items():
            grads[f'self_attn_{key}'] = value

        # Get gradients from cross-attention
        cross_attn_grads = self.cross_attn.get_gradients()
        for key, value in cross_attn_grads.items():
            grads[f'cross_attn_{key}'] = value

        # Get gradients from feed-forward
        ff_grads = self.feed_forward.get_gradients()
        for key, value in ff_grads.items():
            grads[f'ff_{key}'] = value

        # Get gradients from layer norms
        ln1_grads = self.ln1.get_gradients()
        for key, value in ln1_grads.items():
            grads[f'ln1_{key}'] = value

        ln2_grads = self.ln2.get_gradients()
        for key, value in ln2_grads.items():
            grads[f'ln2_{key}'] = value

        ln3_grads = self.ln3.get_gradients()
        for key, value in ln3_grads.items():
            grads[f'ln3_{key}'] = value

        return grads

    def update_params(self, params: Dict):
        """Update parameters in sub-components"""
        # Filter params for each sub-component
        self_attn_params = {}
        cross_attn_params = {}
        ff_params = {}
        ln1_params = {}
        ln2_params = {}
        ln3_params = {}

        for key, value in params.items():
            if key.startswith('self_attn_'):
                self_attn_params[key[10:]] = value
            elif key.startswith('cross_attn_'):
                cross_attn_params[key[11:]] = value
            elif key.startswith('ff_'):
                ff_params[key[3:]] = value
            elif key.startswith('ln1_'):
                ln1_params[key[4:]] = value
            elif key.startswith('ln2_'):
                ln2_params[key[4:]] = value
            elif key.startswith('ln3_'):
                ln3_params[key[4:]] = value

        # Update sub-components
        if self_attn_params:
            self.self_attn.update_params(self_attn_params)
        if cross_attn_params:
            self.cross_attn.update_params(cross_attn_params)
        if ff_params:
            self.feed_forward.update_params(ff_params)
        if ln1_params:
            self.ln1.update_params(ln1_params)
        if ln2_params:
            self.ln2.update_params(ln2_params)
        if ln3_params:
            self.ln3.update_params(ln3_params)


class DecoderStack(Layer):
    """
    Stack of transformer decoder blocks.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward inner dimension
        num_layers (int): Number of decoder blocks in the stack
        dropout (float): Dropout rate
        pre_norm (bool): If True, apply layer norm before attention and feed-forward
        activation (str): Activation function for feed-forward networks
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 pre_norm: bool = True,
                 activation: str = 'relu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pre_norm = pre_norm

        # Initialize list of decoder blocks
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                DecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    pre_norm=pre_norm,
                    activation=activation
                )
            )

        # Final layer norm for pre-norm architecture
        self.ln_final = None
        if pre_norm:
            self.ln_final = LayerNorm(embed_dim)

    def to(self, device_type: str) -> 'DecoderStack':
        """Move layer to specified device"""
        super().to(device_type)
        for layer in self.layers:
            layer.to(device_type)
        if self.ln_final is not None:
            self.ln_final.to(device_type)
        return self

    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        Forward pass through decoder stack.

        Args:
            x: Input tensor of shape (batch_size, tgt_seq_len, embed_dim)
            encoder_output: Output from encoder of shape (batch_size, src_seq_len, embed_dim)
            self_attn_mask: Mask for self-attention to prevent looking at future tokens
            cross_attn_mask: Mask for cross-attention to handle padding in encoder output

        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, embed_dim)
        """
        # Move inputs to correct device
        x = self.device.to_device(x)
        encoder_output = self.device.to_device(encoder_output)
        if self_attn_mask is not None:
            self_attn_mask = self.device.to_device(self_attn_mask)
        if cross_attn_mask is not None:
            cross_attn_mask = self.device.to_device(cross_attn_mask)

        # Store inputs for backward pass
        self.input = x
        self.encoder_output = encoder_output

        # Pass through decoder blocks
        layer_outputs = [x]
        encoder_output_grads = []

        for layer in self.layers:
            x = layer.forward(
                x,
                encoder_output,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask
            )
            layer_outputs.append(x)

        # Apply final layer norm for pre-norm architecture
        if self.pre_norm and self.ln_final is not None:
            x = self.ln_final.forward(x)

        # Store layer outputs for backward pass
        self.layer_outputs = layer_outputs

        return x

    def backward(self, output_gradient):
        """
        Backward pass through decoder stack.

        Args:
            output_gradient: Gradient from next layer

        Returns:
            Tuple of gradients: (input_gradient, encoder_output_gradient)
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        # Apply final layer norm backward if used
        if self.pre_norm and self.ln_final is not None:
            output_gradient = self.ln_final.backward(output_gradient)

        # Initialize encoder output gradient accumulator
        encoder_output_grad = None

        # Backpropagate through decoder blocks in reverse order
        for i in reversed(range(len(self.layers))):
            output_gradient, layer_encoder_grad = self.layers[i].backward(output_gradient)

            # Accumulate encoder output gradients
            if encoder_output_grad is None:
                encoder_output_grad = layer_encoder_grad
            else:
                encoder_output_grad = encoder_output_grad + layer_encoder_grad

        return output_gradient, encoder_output_grad

    def get_trainable_params(self) -> Dict:
        """Get trainable parameters from all layers"""
        params = {}

        # Get parameters from decoder blocks
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_trainable_params()
            for key, value in layer_params.items():
                params[f'layer_{i}_{key}'] = value

        # Get parameters from final layer norm if used
        if self.pre_norm and self.ln_final is not None:
            ln_final_params = self.ln_final.get_trainable_params()
            for key, value in ln_final_params.items():
                params[f'ln_final_{key}'] = value

        return params

    def get_gradients(self) -> Dict:
        """Get parameter gradients from all layers"""
        grads = {}

        # Get gradients from decoder blocks
        for i, layer in enumerate(self.layers):
            layer_grads = layer.get_gradients()
            for key, value in layer_grads.items():
                grads[f'layer_{i}_{key}'] = value

        # Get gradients from final layer norm if used
        if self.pre_norm and self.ln_final is not None:
            ln_final_grads = self.ln_final.get_gradients()
            for key, value in ln_final_grads.items():
                grads[f'ln_final_{key}'] = value

        return grads

    def update_params(self, params: Dict):
        """Update parameters in all layers"""
        # Group parameters by layer
        layer_params = [dict() for _ in range(len(self.layers))]
        ln_final_params = {}

        for key, value in params.items():
            if key.startswith('layer_'):
                # Extract layer index and parameter name
                parts = key.split('_', 2)
                if len(parts) >= 3:
                    layer_idx = int(parts[1])
                    param_name = parts[2]
                    if 0 <= layer_idx < len(self.layers):
                        layer_params[layer_idx][param_name] = value
            elif key.startswith('ln_final_'):
                param_name = key[9:]  # Remove 'ln_final_' prefix
                ln_final_params[param_name] = value

        # Update decoder blocks
        for i, layer in enumerate(self.layers):
            if layer_params[i]:
                layer.update_params(layer_params[i])

        # Update final layer norm if used
        if self.pre_norm and self.ln_final is not None and ln_final_params:
            self.ln_final.update_params(ln_final_params)