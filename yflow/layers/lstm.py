import numpy as np
from typing import Dict, List, Optional, Union
from ..core.layer import Layer
from ..core.device import Device
from .normalization import LayerNorm
class YSTM(Layer):
    """
    YSTM (LSTM) layer with GPU support and proper shape handling
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: Optional[int] = None,
                 layer_norm: bool = True,
                 dropout: float = 0.0,
                 return_sequences: bool = False,
                 batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        self.use_layer_norm = layer_norm
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.batch_first = batch_first
        self.initialized = False

        # Initialize weights
        self._init_weights()

        # Initialize LayerNorm if needed
        if self.use_layer_norm:
            self.layer_norm = LayerNorm(hidden_size)

        self.reset_states()

    def _init_weights(self):
        """Initialize gates with GPU support"""
        xp = self.device.xp
        scale = 1.0 / xp.sqrt(self.hidden_size)

        # Input gate
        self.W_ii = self.device.to_device(xp.random.uniform(-scale, scale, (self.input_size, self.hidden_size)))
        self.W_hi = self.device.to_device(xp.random.uniform(-scale, scale, (self.hidden_size, self.hidden_size)))
        self.b_i = self.device.to_device(xp.zeros(self.hidden_size))

        # Forget gate
        self.W_if = self.device.to_device(xp.random.uniform(-scale, scale, (self.input_size, self.hidden_size)))
        self.W_hf = self.device.to_device(xp.random.uniform(-scale, scale, (self.hidden_size, self.hidden_size)))
        self.b_f = self.device.to_device(xp.ones(self.hidden_size))  # Bias towards remembering

        # Cell gate
        self.W_ig = self.device.to_device(xp.random.uniform(-scale, scale, (self.input_size, self.hidden_size)))
        self.W_hg = self.device.to_device(xp.random.uniform(-scale, scale, (self.hidden_size, self.hidden_size)))
        self.b_g = self.device.to_device(xp.zeros(self.hidden_size))

        # Output gate
        self.W_io = self.device.to_device(xp.random.uniform(-scale, scale, (self.input_size, self.hidden_size)))
        self.W_ho = self.device.to_device(xp.random.uniform(-scale, scale, (self.hidden_size, self.hidden_size)))
        self.b_o = self.device.to_device(xp.zeros(self.hidden_size))

    def reset_states(self):
        """Reset hidden and cell states"""
        self.h = None
        self.c = None
        self.cache = {}
        if hasattr(self, 'layer_norm'):
            self.layer_norm.cache = {}

    def get_expected_input_shape(self) -> tuple:
        """Get expected input shape with None for flexible dimensions"""
        if self.batch_first:
            return (None, None, self.input_size)  # (batch_size, seq_length, features)
        return (None, None, self.input_size)  # (seq_length, batch_size, features)

    def _sigmoid(self, x):
        """Numerically stable sigmoid with GPU support"""
        xp = self.device.xp
        return 1.0 / (1.0 + xp.exp(-xp.clip(x, -709, 709)))

    def _apply_dropout(self, x):
        """Apply dropout during training with GPU support"""
        if self.training and self.dropout > 0:
            xp = self.device.xp
            mask = (xp.random.rand(*x.shape) > self.dropout) / (1 - self.dropout)
            return x * mask
        return x

    def forward(self, x: Union[np.ndarray, List[np.ndarray]]):
        """Forward pass with GPU support"""
        # Handle list input and ensure it's on the correct device
        if isinstance(x, list):
            x = self.device.to_device(np.array(x))
        else:
            x = self.device.to_device(x)

        # Ensure 3D input
        if x.ndim == 2:
            x = x.reshape(1, *x.shape)

        batch_size, seq_length, _ = x.shape
        xp = self.device.xp

        # Initialize states based on current batch size
        if self.h is None or self.h.shape[0] != batch_size:
            self.h = self.device.to_device(xp.zeros((batch_size, self.hidden_size)))
            self.c = self.device.to_device(xp.zeros((batch_size, self.hidden_size)))

        # Storage for sequence outputs and cache
        outputs = []
        self.cache = {
            'gates': [],
            'states': [],
            'x': x
        }

        # Process sequence
        for t in range(seq_length):
            x_t = self.device.to_device(x[:, t, :])

            # Ensure weights are on the correct device
            W_ii = self.device.to_device(self.W_ii)
            W_hi = self.device.to_device(self.W_hi)
            W_if = self.device.to_device(self.W_if)
            W_hf = self.device.to_device(self.W_hf)
            W_ig = self.device.to_device(self.W_ig)
            W_hg = self.device.to_device(self.W_hg)
            W_io = self.device.to_device(self.W_io)
            W_ho = self.device.to_device(self.W_ho)

            # Input gate
            i_t = self._sigmoid(
                xp.dot(x_t, W_ii) +
                xp.dot(self.h, W_hi) +
                self.b_i
            )

            # Forget gate
            f_t = self._sigmoid(
                xp.dot(x_t, W_if) +
                xp.dot(self.h, W_hf) +
                self.b_f
            )

            # Cell gate
            g_t = xp.tanh(
                xp.dot(x_t, W_ig) +
                xp.dot(self.h, W_hg) +
                self.b_g
            )

            # Output gate
            o_t = self._sigmoid(
                xp.dot(x_t, W_io) +
                xp.dot(self.h, W_ho) +
                self.b_o
            )

            # Update cell state
            self.c = f_t * self.c + i_t * g_t

            # Update hidden state
            h_raw = o_t * xp.tanh(self.c)

            # Apply layer normalization if enabled
            if self.use_layer_norm:
                self.h = self.layer_norm.forward(h_raw)
            else:
                self.h = h_raw

            # Apply dropout
            self.h = self._apply_dropout(self.h)

            # Store states and gates for backward pass
            self.cache['gates'].append((i_t, f_t, g_t, o_t))
            self.cache['states'].append((self.c.copy(), self.h.copy()))

            # Store output
            outputs.append(self.h)

        # Stack outputs
        outputs = xp.stack(outputs, axis=1)

        # Return either full sequence or just last state
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]

    def backward(self, output_gradient):
        """Backward pass with GPU support"""
        output_gradient = self.device.to_device(output_gradient)
        x = self.cache['x']
        batch_size, seq_length, _ = x.shape
        xp = self.device.xp

        # Initialize gradients on device
        dW_ii = xp.zeros_like(self.W_ii)
        dW_hi = xp.zeros_like(self.W_hi)
        db_i = xp.zeros_like(self.b_i)

        dW_if = xp.zeros_like(self.W_if)
        dW_hf = xp.zeros_like(self.W_hf)
        db_f = xp.zeros_like(self.b_f)

        dW_ig = xp.zeros_like(self.W_ig)
        dW_hg = xp.zeros_like(self.W_hg)
        db_g = xp.zeros_like(self.b_g)

        dW_io = xp.zeros_like(self.W_io)
        dW_ho = xp.zeros_like(self.W_ho)
        db_o = xp.zeros_like(self.b_o)

        # Initialize gradient flows
        dh_next = xp.zeros((batch_size, self.hidden_size))
        dc_next = xp.zeros((batch_size, self.hidden_size))

        # Handle output gradient shape for non-sequence return
        if not self.return_sequences:
            full_gradient = xp.zeros((batch_size, seq_length, self.hidden_size))
            full_gradient[:, -1, :] = output_gradient
            output_gradient = full_gradient

        # Gradients for input sequence
        dx = xp.zeros_like(x)

        # Backward through time
        for t in reversed(range(seq_length)):
            i_t, f_t, g_t, o_t = self.cache['gates'][t]
            c_t, h_t = self.cache['states'][t]
            x_t = x[:, t, :]

            # Get previous cell state
            if t > 0:
                c_prev = self.cache['states'][t - 1][0]
            else:
                c_prev = xp.zeros_like(c_t)

            # Get upstream gradient for this timestep
            dh = output_gradient[:, t, :] + dh_next

            # Apply layer norm gradient if used
            if self.use_layer_norm:
                dh = self.layer_norm.backward(dh)

            # Output gate
            do = dh * xp.tanh(c_t)
            do = do * o_t * (1 - o_t)

            # Cell state
            dc = dh * o_t * (1 - xp.tanh(c_t) ** 2)
            dc = dc + dc_next

            # Input gate
            di = dc * g_t
            di = di * i_t * (1 - i_t)

            # Forget gate
            df = dc * c_prev
            df = df * f_t * (1 - f_t)

            # Cell gate
            dg = dc * i_t
            dg = dg * (1 - g_t ** 2)

            # Input weights
            dW_ii += xp.dot(x_t.T, di) / batch_size
            dW_if += xp.dot(x_t.T, df) / batch_size
            dW_ig += xp.dot(x_t.T, dg) / batch_size
            dW_io += xp.dot(x_t.T, do) / batch_size

            # Hidden weights
            if t > 0:
                h_prev = self.cache['states'][t - 1][1]
            else:
                h_prev = xp.zeros_like(h_t)

            dW_hi += xp.dot(h_prev.T, di) / batch_size
            dW_hf += xp.dot(h_prev.T, df) / batch_size
            dW_hg += xp.dot(h_prev.T, dg) / batch_size
            dW_ho += xp.dot(h_prev.T, do) / batch_size

            # Biases
            db_i += xp.sum(di, axis=0) / batch_size
            db_f += xp.sum(df, axis=0) / batch_size
            db_g += xp.sum(dg, axis=0) / batch_size
            db_o += xp.sum(do, axis=0) / batch_size

            # Gradient for input
            dx[:, t, :] = (xp.dot(di, self.W_ii.T) +
                          xp.dot(df, self.W_if.T) +
                          xp.dot(dg, self.W_ig.T) +
                          xp.dot(do, self.W_io.T))

            # Gradient for next timestep
            dh_next = (xp.dot(di, self.W_hi.T) +
                      xp.dot(df, self.W_hf.T) +
                      xp.dot(dg, self.W_hg.T) +
                      xp.dot(do, self.W_ho.T))

            dc_next = dc * f_t

        # Store gradients
        self.gradients = {
            'W_ii': dW_ii, 'W_hi': dW_hi, 'b_i': db_i,
            'W_if': dW_if, 'W_hf': dW_hf, 'b_f': db_f,
            'W_ig': dW_ig, 'W_hg': dW_hg, 'b_g': db_g,
            'W_io': dW_io, 'W_ho': dW_ho, 'b_o': db_o
        }

        return dx

    def get_trainable_params(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters"""
        params = {
            'W_ii': self.W_ii, 'W_hi': self.W_hi, 'b_i': self.b_i,
            'W_if': self.W_if, 'W_hf': self.W_hf, 'b_f': self.b_f,
            'W_ig': self.W_ig, 'W_hg': self.W_hg, 'b_g': self.b_g,
            'W_io': self.W_io, 'W_ho': self.W_ho, 'b_o': self.b_o
        }
        if self.use_layer_norm:
            params.update(self.layer_norm.get_trainable_params())
        return params

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients"""
        grads = self.gradients.copy()
        if self.use_layer_norm:
            grads.update(self.layer_norm.get_gradients())
        return grads

    def update_params(self, params: Dict[str, np.ndarray]):
        """Update layer parameters"""
        for name, param in params.items():
            if hasattr(self, name):
                setattr(self, name, self.device.to_device(param))
        if self.use_layer_norm:
            ln_params = {k: v for k, v in params.items()
                         if k in ['gamma', 'beta']}
            if ln_params:
                self.layer_norm.update_params(ln_params)

    def get_config(self) -> dict:
        """Get layer configuration"""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'layer_norm': self.use_layer_norm,
            'dropout': self.dropout,
            'return_sequences': self.return_sequences,
            'batch_first': self.batch_first
        }

    @classmethod
    def from_config(cls, config: dict) -> 'YSTM':
        """Create layer from configuration"""
        return cls(**config)