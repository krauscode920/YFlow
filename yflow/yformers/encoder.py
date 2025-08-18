# yflow/yformers/encoder.py
"""
Encoder components for transformer models.

This module implements the encoder blocks and stacks used in transformer architectures.
All implementations leverage YFlow's device abstraction for seamless CPU/GPU support.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, List
from ..core.layer import Layer
from ..core.device import Device
from .attention import MultiHeadAttention
from ..layers.normalization import LayerNorm
from ..layers.dense import Dense
from ..layers.activations import ReLU, GELU



class FeedForward(Layer):
    """
    Feed-forward network used in transformer encoder and decoder blocks.

    Consists of two linear transformations with a nonlinearity in between.

    Args:
        embed_dim (int): Input and output dimension
        ff_dim (int): Dimension of the inner feed-forward layer
        dropout (float): Dropout rate
        activation (str): Activation function to use ('relu' or 'gelu')
    """

    def __init__(self,
                 embed_dim: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.training = True

        # Initialize layers
        self.fc1 = Dense(
            input_size=embed_dim,
            output_size=ff_dim
        )

        # Set activation function
        if activation.lower() == 'relu':
            self.activation = ReLU()
        elif activation.lower() == 'gelu':
            self.activation = GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'relu' or 'gelu'.")

        self.fc2 = Dense(
            input_size=ff_dim,
            output_size=embed_dim
        )

    def to(self, device_type: str) -> 'FeedForward':
        """Move layer to specified device"""
        super().to(device_type)
        self.fc1.to(device_type)
        self.activation.to(device_type)
        self.fc2.to(device_type)
        return self

    def _apply_dropout(self, x):
        """Apply dropout during training"""
        xp = self.device.xp
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            dropout_mask = xp.random.binomial(1, keep_prob, x.shape) / keep_prob
            return x * dropout_mask
        return x

    # Replace the forward method in yflow/yformers/encoder.py FeedForward class:

    def forward(self, x):
        """
        Forward pass for feed-forward network.
        Handles 3D inputs properly for transformers.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Move input to correct device
        x = self.device.to_device(x)

        # Store input for backward pass
        self.input = x
        original_shape = x.shape

        # Reshape to 2D for dense layers: (batch_size * seq_len, embed_dim)
        batch_size, seq_len, embed_dim = original_shape
        x_reshaped = x.reshape(-1, embed_dim)

        # First linear transformation
        hidden = self.fc1.forward(x_reshaped)

        # Apply activation
        activated = self.activation.forward(hidden)

        # Apply dropout
        activated_dropped = self._apply_dropout(activated)
        self.dropout_mask1 = getattr(self, 'dropout_mask', None)

        # Second linear transformation
        output = self.fc2.forward(activated_dropped)

        # Apply dropout
        output_dropped = self._apply_dropout(output)
        self.dropout_mask2 = getattr(self, 'dropout_mask', None)

        # Reshape back to 3D: (batch_size, seq_len, embed_dim)
        output_3d = output_dropped.reshape(batch_size, seq_len, embed_dim)

        return output_3d

    # Also replace the backward method in yflow/yformers/encoder.py FeedForward class:

    def backward(self, output_gradient):
        """
        Backward pass for feed-forward network.
        Handles 3D gradients properly for transformers.

        Args:
            output_gradient: Gradient from next layer of shape (batch_size, seq_len, embed_dim)

        Returns:
            Gradient with respect to input of shape (batch_size, seq_len, embed_dim)
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        # Get original shape from stored input
        batch_size, seq_len, embed_dim = self.input.shape

        # Reshape gradient to 2D: (batch_size * seq_len, embed_dim)
        output_grad_2d = output_gradient.reshape(-1, embed_dim)

        # Apply output dropout gradient if used
        if self.training and self.dropout_rate > 0 and self.dropout_mask2 is not None:
            dropout_mask_2d = self.dropout_mask2.reshape(-1, embed_dim)
            output_grad_2d = output_grad_2d * dropout_mask_2d

        # Backpropagate through second linear layer
        activated_grad = self.fc2.backward(output_grad_2d)

        # Apply activation dropout gradient if used
        if self.training and self.dropout_rate > 0 and self.dropout_mask1 is not None:
            dropout_mask_1d = self.dropout_mask1.reshape(-1, self.ff_dim)
            activated_grad = activated_grad * dropout_mask_1d

        # Backpropagate through activation
        hidden_grad = self.activation.backward(activated_grad)

        # Backpropagate through first linear layer
        input_grad_2d = self.fc1.backward(hidden_grad)

        # Reshape back to 3D: (batch_size, seq_len, embed_dim)
        input_grad_3d = input_grad_2d.reshape(batch_size, seq_len, embed_dim)

        return input_grad_3d


    def get_trainable_params(self) -> Dict:
        """Get trainable parameters from sub-layers"""
        params = {}

        # Get parameters from fc1
        fc1_params = self.fc1.get_trainable_params()
        for key, value in fc1_params.items():
            params[f'fc1_{key}'] = value

        # Get parameters from fc2
        fc2_params = self.fc2.get_trainable_params()
        for key, value in fc2_params.items():
            params[f'fc2_{key}'] = value

        return params

    def get_gradients(self) -> Dict:
        """Get parameter gradients from sub-layers"""
        grads = {}

        # Get gradients from fc1
        fc1_grads = self.fc1.get_gradients()
        for key, value in fc1_grads.items():
            grads[f'fc1_{key}'] = value

        # Get gradients from fc2
        fc2_grads = self.fc2.get_gradients()
        for key, value in fc2_grads.items():
            grads[f'fc2_{key}'] = value

        return grads

    def update_params(self, params: Dict):
        """Update parameters in sub-layers"""
        # Filter params for each sub-layer
        fc1_params = {}
        fc2_params = {}

        for key, value in params.items():
            if key.startswith('fc1_'):
                fc1_params[key[4:]] = value
            elif key.startswith('fc2_'):
                fc2_params[key[4:]] = value

        # Update sub-layers
        if fc1_params:
            self.fc1.update_params(fc1_params)
        if fc2_params:
            self.fc2.update_params(fc2_params)


class EncoderBlock(Layer):
    """
    Transformer encoder block.

    Each block consists of:
    1. Multi-head self-attention with residual connection and layer norm
    2. Feed-forward network with residual connection and layer norm

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
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.ln1 = LayerNorm(self.embed_dim)

        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation
        )

        self.ln2 = LayerNorm(self.embed_dim)

    def to(self, device_type: str) -> 'EncoderBlock':
        """Move layer to specified device"""
        super().to(device_type)
        self.self_attn.to(device_type)
        self.ln1.to(device_type)
        self.feed_forward.to(device_type)
        self.ln2.to(device_type)
        return self

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
        Forward pass for encoder block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Attention mask of shape (batch_size, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Move input to correct device
        x = self.device.to_device(x)

        # Store input for backward pass
        self.input = x

        # Attention with residual connection
        if self.pre_norm:
            # Pre-LN architecture
            norm1 = self.ln1.forward(x)
            attn_output = self.self_attn.forward(norm1, mask=mask)
            attn_output = self._apply_dropout(attn_output)
            self.attn_dropout_mask = getattr(self, 'dropout_mask', None)
            x1 = x + attn_output

            # Feed-forward with residual connection
            norm2 = self.ln2.forward(x1)
            ff_output = self.feed_forward.forward(norm2)
            ff_output = self._apply_dropout(ff_output)
            self.ff_dropout_mask = getattr(self, 'dropout_mask', None)
            output = x1 + ff_output
        else:
            # Post-LN architecture
            attn_output = self.self_attn.forward(x, mask=mask)
            attn_output = self._apply_dropout(attn_output)
            self.attn_dropout_mask = getattr(self, 'dropout_mask', None)
            x1 = self.ln1.forward(x + attn_output)

            # Feed-forward with residual connection
            ff_output = self.feed_forward.forward(x1)
            ff_output = self._apply_dropout(ff_output)
            self.ff_dropout_mask = getattr(self, 'dropout_mask', None)
            output = self.ln2.forward(x1 + ff_output)

        # Save intermediate outputs for backward pass
        self.attn_output = attn_output
        self.ff_output = ff_output
        self.x1 = x1

        return output

    def backward(self, output_gradient):
        """
        Backward pass for encoder block.

        Args:
            output_gradient: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        if self.pre_norm:
            # Pre-LN architecture (working backwards)
            # Gradient through residual connection
            ff_grad = output_gradient
            x1_grad = output_gradient

            # Apply feed-forward dropout gradient if used
            if self.training and self.dropout_rate > 0 and self.ff_dropout_mask is not None:
                ff_grad = ff_grad * self.ff_dropout_mask

            # Backpropagate through feed-forward
            norm2_grad = self.feed_forward.backward(ff_grad)

            # Backpropagate through layer norm
            x1_grad2 = self.ln2.backward(norm2_grad)

            # Sum gradients from residual connection
            x1_grad = x1_grad + x1_grad2

            # Gradient through residual connection
            attn_grad = x1_grad
            x_grad = x1_grad

            # Apply attention dropout gradient if used
            if self.training and self.dropout_rate > 0 and self.attn_dropout_mask is not None:
                attn_grad = attn_grad * self.attn_dropout_mask

            # Backpropagate through attention
            norm1_grad = self.self_attn.backward(attn_grad)

            # Backpropagate through layer norm
            x_grad2 = self.ln1.backward(norm1_grad)

            # Sum gradients from residual connection
            x_grad = x_grad + x_grad2

        else:
            # Post-LN architecture (working backwards)
            # Backpropagate through second layer norm
            x1_plus_ff_grad = self.ln2.backward(output_gradient)

            # Gradient through residual connection
            ff_grad = x1_plus_ff_grad
            x1_grad = x1_plus_ff_grad

            # Apply feed-forward dropout gradient if used
            if self.training and self.dropout_rate > 0 and self.ff_dropout_mask is not None:
                ff_grad = ff_grad * self.ff_dropout_mask

            # Backpropagate through feed-forward
            x1_grad2 = self.feed_forward.backward(ff_grad)

            # Sum gradients at x1
            x1_grad = x1_grad + x1_grad2

            # Backpropagate through first layer norm
            x_plus_attn_grad = self.ln1.backward(x1_grad)

            # Gradient through residual connection
            attn_grad = x_plus_attn_grad
            x_grad = x_plus_attn_grad

            # Apply attention dropout gradient if used
            if self.training and self.dropout_rate > 0 and self.attn_dropout_mask is not None:
                attn_grad = attn_grad * self.attn_dropout_mask

            # Backpropagate through attention
            x_grad2 = self.self_attn.backward(attn_grad)

            # Sum gradients at x
            x_grad = x_grad + x_grad2

        return x_grad

    def get_trainable_params(self) -> Dict:
        """Get trainable parameters from sub-components"""
        params = {}

        # Get parameters from attention
        attn_params = self.self_attn.get_trainable_params()
        for key, value in attn_params.items():
            params[f'attn_{key}'] = value

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

        return params

    def get_gradients(self) -> Dict:
        """Get parameter gradients from sub-components"""
        grads = {}

        # Get gradients from attention
        attn_grads = self.self_attn.get_gradients()
        for key, value in attn_grads.items():
            grads[f'attn_{key}'] = value

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

        return grads

    def update_params(self, params: Dict):
        """Update parameters in sub-components"""
        # Filter params for each sub-component
        attn_params = {}
        ff_params = {}
        ln1_params = {}
        ln2_params = {}

        for key, value in params.items():
            if key.startswith('attn_'):
                attn_params[key[5:]] = value
            elif key.startswith('ff_'):
                ff_params[key[3:]] = value
            elif key.startswith('ln1_'):
                ln1_params[key[4:]] = value
            elif key.startswith('ln2_'):
                ln2_params[key[4:]] = value

        # Update sub-components
        if attn_params:
            self.self_attn.update_params(attn_params)
        if ff_params:
            self.feed_forward.update_params(ff_params)
        if ln1_params:
            self.ln1.update_params(ln1_params)
        if ln2_params:
            self.ln2.update_params(ln2_params)


class EncoderStack(Layer):
    """
    Stack of transformer encoder blocks.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward inner dimension
        num_layers (int): Number of encoder blocks in the stack
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

        # Initialize list of encoder blocks
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                EncoderBlock(
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

    def to(self, device_type: str) -> 'EncoderStack':
        """Move layer to specified device"""
        super().to(device_type)
        for layer in self.layers:
            layer.to(device_type)
        if self.ln_final is not None:
            self.ln_final.to(device_type)
        return self

    def forward(self, x, mask=None):
        """
        Forward pass through encoder stack.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Attention mask of shape (batch_size, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Move input to correct device
        x = self.device.to_device(x)
        if mask is not None:
            mask = self.device.to_device(mask)

        # Store input for backward pass
        self.input = x

        # Pass through encoder blocks
        layer_outputs = [x]
        for layer in self.layers:
            x = layer.forward(x, mask)
            layer_outputs.append(x)

        # Apply final layer norm for pre-norm architecture
        if self.pre_norm and self.ln_final is not None:
            x = self.ln_final.forward(x)

        # Store layer outputs for backward pass
        self.layer_outputs = layer_outputs

        return x

    def backward(self, output_gradient):
        """
        Backward pass through encoder stack.

        Args:
            output_gradient: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        # Ensure gradient is on correct device
        output_gradient = self.device.to_device(output_gradient)

        # Apply final layer norm backward if used
        if self.pre_norm and self.ln_final is not None:
            output_gradient = self.ln_final.backward(output_gradient)

        # Backpropagate through encoder blocks in reverse order
        for i in reversed(range(len(self.layers))):
            output_gradient = self.layers[i].backward(output_gradient)

        return output_gradient

    def get_trainable_params(self) -> Dict:
        """Get trainable parameters from all layers"""
        params = {}

        # Get parameters from encoder blocks
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

        # Get gradients from encoder blocks
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

        # Update encoder blocks
        for i, layer in enumerate(self.layers):
            if layer_params[i]:
                layer.update_params(layer_params[i])

        # Update final layer norm if used
        if self.pre_norm and self.ln_final is not None and ln_final_params:
            self.ln_final.update_params(ln_final_params)