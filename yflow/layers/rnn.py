import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from ..core.layer import Layer, _set_global_seed
from ..core.device import Device


class GradientTracker:
    """Utility class to track and analyze gradients with GPU support"""

    def __init__(self, clip_value: Optional[float] = None):
        self.clip_value = clip_value
        self.history: List[float] = []

    def track(self, gradient, name: str = "") -> Union[np.ndarray, 'cp.ndarray']:
        """Track and optionally clip gradient values"""
        xp = self.device.xp if hasattr(self, 'device') else np

        if xp.any(xp.isnan(gradient)) or xp.any(xp.isinf(gradient)):
            raise ValueError(f"Invalid values in {name}: contains NaN or Inf")

        if self.clip_value is not None:
            norm = xp.linalg.norm(gradient)
            if norm > self.clip_value:
                gradient = gradient * (self.clip_value / norm)

        self.history.append(float(xp.linalg.norm(gradient)))
        return gradient

    def get_stats(self) -> Dict[str, float]:
        if not self.history:
            return {}
        return {
            'mean': float(np.mean(self.history)),
            'std': float(np.std(self.history)),
            'max': float(np.max(self.history)),
            'min': float(np.min(self.history))
        }


class YQuence(Layer):
    """Enhanced RNN with variable length sequence support and GPU acceleration"""

    def __init__(self,
                 input_size: Optional[int] = None,
                 hidden_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 activation: str = 'tanh',
                 regularization: Optional[str] = None,
                 reg_strength: float = 0.01,
                 return_sequences: bool = False,
                 batch_first: bool = True,
                 gradient_clipping: Optional[float] = None,
                 dropout: float = 0.0,
                 padding: str = 'post',
                 mask_zero: bool = True):
        super().__init__()
        _set_global_seed()

        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'activation': activation,
            'regularization': regularization,
            'reg_strength': reg_strength,
            'return_sequences': return_sequences,
            'batch_first': batch_first,
            'gradient_clipping': gradient_clipping,
            'dropout': dropout,
            'padding': padding,
            'mask_zero': mask_zero
        }

        self._validate_config()
        self.initialized = False
        self.mask = None
        self.hidden_state = None
        self.gradient_tracker = GradientTracker(gradient_clipping)
        self.cache = {}

    def _prepare_sequences(self, sequences: Union[np.ndarray, List[np.ndarray]]) -> Tuple[
        Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']]:
        """Prepare sequences and generate masks with GPU support"""
        xp = self.device.xp

        if isinstance(sequences, list):
            max_length = max(len(seq) for seq in sequences)
            padded_sequences = []
            mask = []

            for seq in sequences:
                if seq.ndim == 1:
                    seq = seq.reshape(-1, 1)

                seq_mask = xp.ones((max_length,))
                seq_mask[len(seq):] = 0

                if self.config['padding'] == 'post':
                    padded_seq = xp.zeros((max_length, seq.shape[1]))
                    padded_seq[:len(seq)] = self.device.to_device(seq)
                else:  # pre-padding
                    padded_seq = xp.zeros((max_length, seq.shape[1]))
                    padded_seq[-len(seq):] = self.device.to_device(seq)
                    seq_mask = xp.roll(seq_mask, max_length - len(seq))

                padded_sequences.append(padded_seq)
                mask.append(seq_mask)

            padded_sequences = xp.stack(padded_sequences)
            mask = xp.stack(mask)
        else:
            sequences = self.device.to_device(sequences)
            if self.config['mask_zero']:
                mask = (sequences != 0).any(axis=2) if sequences.ndim == 3 else (sequences != 0)
            else:
                mask = xp.ones(sequences.shape[:-1] if sequences.ndim == 3 else sequences.shape)
            padded_sequences = sequences

        return padded_sequences, mask.astype(bool)

    def _initialize_parameters(self):
        """Initialize weights with GPU support"""
        xp = self.device.xp
        input_size = self.config['input_size']
        hidden_size = self.config['hidden_size']
        output_size = self.config['output_size']

        limit_xh = xp.sqrt(2.0 / (input_size + hidden_size))
        limit_hh = xp.sqrt(2.0 / (hidden_size + hidden_size))
        limit_hy = xp.sqrt(2.0 / (hidden_size + output_size))

        self.W_x = self.device.to_device(
            xp.random.uniform(-limit_xh, limit_xh, (input_size, hidden_size))
        )
        self.W_h = self.device.to_device(
            xp.random.uniform(-limit_hh, limit_hh, (hidden_size, hidden_size))
        )
        self.W_y = self.device.to_device(
            xp.random.uniform(-limit_hy, limit_hy, (hidden_size, output_size))
        )

        self.b_h = self.device.to_device(xp.zeros((1, hidden_size)))
        self.b_y = self.device.to_device(xp.zeros((1, output_size)))

        self.initialized = True

    def _activate(self, x: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Apply activation function with GPU support"""
        xp = self.device.xp

        if self.config['activation'] == 'tanh':
            return xp.tanh(x)
        elif self.config['activation'] == 'relu':
            return xp.maximum(0, x)
        else:  # sigmoid
            return 1 / (1 + xp.exp(-xp.clip(x, -709, 709)))

    def _activate_derivative(self, x: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Calculate activation function derivative with GPU support"""
        xp = self.device.xp

        if self.config['activation'] == 'tanh':
            return 1 - x ** 2
        elif self.config['activation'] == 'relu':
            return (x > 0).astype(xp.float32)
        else:  # sigmoid
            return x * (1 - x)

    def forward(self, x: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, 'cp.ndarray']:
        """Forward pass with mask support and GPU optimization"""
        x, self.mask = self._prepare_sequences(x)

        if x.ndim != 3:
            x = self.shape_handler.auto_reshape(x, target_ndim=3)
            if self.mask.ndim != 2:
                self.mask = self.mask.reshape(x.shape[0], x.shape[1])

        if not self.initialized:
            self._infer_shapes(x.shape)
            self._initialize_parameters()

        if self.config['batch_first']:
            x = self.device.xp.transpose(x, (1, 0, 2))
            self.mask = self.device.xp.transpose(self.mask)

        seq_length, batch_size, input_features = x.shape

        if self.hidden_state is None or self.hidden_state.shape[0] != batch_size:
            self.hidden_state = self.device.xp.zeros((batch_size, self.config['hidden_size']))

        hidden_states = []
        outputs = []
        hidden_inputs = []
        xp = self.device.xp

        for t in range(seq_length):
            x_t = x[t]
            mask_t = self.mask[t] if self.mask is not None else xp.ones(batch_size)

            Wx_out = xp.dot(x_t, self.W_x)
            Wh_out = xp.dot(self.hidden_state, self.W_h)
            hidden_input = Wx_out + Wh_out + self.b_h
            hidden_inputs.append(hidden_input)

            new_hidden = self._activate(hidden_input)
            mask_t = mask_t.reshape(-1, 1)
            self.hidden_state = mask_t * new_hidden + (1 - mask_t) * self.hidden_state

            if self.training and self.config['dropout'] > 0:
                dropout_mask = xp.random.binomial(1, 1 - self.config['dropout'],
                                                  self.hidden_state.shape) / (1 - self.config['dropout'])
                self.hidden_state *= dropout_mask

            current_output = xp.dot(self.hidden_state, self.W_y) + self.b_y
            hidden_states.append(self.hidden_state)
            outputs.append(current_output)

        outputs = xp.stack(outputs)

        self.cache.update({
            'x': x,
            'hidden_states': xp.stack(hidden_states),
            'hidden_inputs': hidden_inputs,
            'seq_length': seq_length,
            'batch_size': batch_size,
            'mask': self.mask
        })

        if self.config['return_sequences']:
            return outputs.transpose(1, 0, 2) if self.config['batch_first'] else outputs
        return outputs[-1]

    def backward(self, output_gradient: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Backward pass with GPU support"""
        xp = self.device.xp
        x = self.cache['x']
        hidden_states = self.cache['hidden_states']
        hidden_inputs = self.cache['hidden_inputs']
        seq_length = self.cache['seq_length']
        batch_size = self.cache['batch_size']
        mask = self.cache['mask']

        dW_x = xp.zeros_like(self.W_x)
        dW_h = xp.zeros_like(self.W_h)
        dW_y = xp.zeros_like(self.W_y)
        db_h = xp.zeros_like(self.b_h)
        db_y = xp.zeros_like(self.b_y)

        if not self.config['return_sequences']:
            output_gradient = output_gradient.reshape(batch_size, -1)

        next_hidden_grad = xp.zeros((batch_size, self.config['hidden_size']))
        next_layer_grad = xp.zeros((seq_length, batch_size, self.config['input_size']))

        for t in reversed(range(seq_length)):
            mask_t = mask[t] if mask is not None else xp.ones(batch_size)
            mask_t = mask_t.reshape(-1, 1)

            x_t = x[t]
            h_t = hidden_states[t]
            hidden_input = hidden_inputs[t]

            if self.config['return_sequences']:
                dy_t = output_gradient[t] if self.config['batch_first'] else output_gradient[:, t]
            else:
                dy_t = output_gradient if t == seq_length - 1 else xp.zeros((batch_size, self.config['output_size']))

            dy_t = dy_t * mask_t

            dW_y += (1.0 / batch_size) * xp.dot(h_t.T, dy_t)
            db_y += (1.0 / batch_size) * xp.sum(dy_t, axis=0, keepdims=True)

            dh = xp.dot(dy_t, self.W_y.T) + next_hidden_grad
            dh = dh * mask_t

            dh_raw = dh * self._activate_derivative(self._activate(hidden_input))

            if t > 0:
                dW_h += (1.0 / batch_size) * xp.dot(hidden_states[t - 1].T, dh_raw)
            db_h += (1.0 / batch_size) * xp.sum(dh_raw, axis=0, keepdims=True)
            dW_x += (1.0 / batch_size) * xp.dot(x_t.T, dh_raw)

            next_hidden_grad = xp.dot(dh_raw, self.W_h.T)
            next_layer_grad[t] = xp.dot(dh_raw, self.W_x.T)

        if self.config['regularization']:
            dW_x += self._regularization_gradient(self.W_x)
            dW_h += self._regularization_gradient(self.W_h)
            dW_y += self._regularization_gradient(self.W_y)

        self.gradients = {
            'W_x': self.gradient_tracker.track(dW_x, 'W_x'),
            'W_h': self.gradient_tracker.track(dW_h, 'W_h'),
            'W_y': self.gradient_tracker.track(dW_y, 'W_y'),
            'b_h': self.gradient_tracker.track(db_h, 'b_h'),
            'b_y': self.gradient_tracker.track(db_y, 'b_y')
        }

        return next_layer_grad.transpose(1, 0, 2) if self.config['batch_first'] else next_layer_grad

    def _validate_config(self):
        """Validate configuration parameters"""
        super()._validate_config()

        valid_activations = ['tanh', 'relu', 'sigmoid']
        if self.config['activation'] not in valid_activations:
            raise ValueError(f"Activation must be one of {valid_activations}")

        valid_padding = ['pre', 'post']
        if self.config['padding'] not in valid_padding:
            raise ValueError(f"Padding must be one of {valid_padding}")

        if not 0 <= self.config['dropout'] < 1:
            raise ValueError("Dropout must be between 0 and 1")

        if self.config['regularization'] is not None:
            valid_regularization = ['l1', 'l2']
            if self.config['regularization'] not in valid_regularization:
                raise ValueError(f"Regularization must be one of {valid_regularization}")

            if self.config['reg_strength'] <= 0:
                raise ValueError("Regularization strength must be positive")

    def get_mask(self) -> Optional[Union[np.ndarray, 'cp.ndarray']]:
        """Return current mask if any"""
        return self.mask


class BiYQuence(Layer):
    """
    Bidirectional wrapper for YQuence layer with GPU support and improved gradient flow.

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden state
        output_size: Size of output
        merge_mode: How to combine outputs ('concat', 'sum', 'mul', 'ave')
        activation: Activation function
        return_sequences: Whether to return full sequence or just last output
    """

    def __init__(self,
                 input_size: Optional[int] = None,
                 hidden_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 merge_mode: str = 'concat',
                 activation: str = 'tanh',
                 return_sequences: bool = False,
                 **kwargs):
        super().__init__()

        valid_merge_modes = ['concat', 'sum', 'mul', 'ave']
        if merge_mode not in valid_merge_modes:
            raise ValueError(f"merge_mode must be one of {valid_merge_modes}")

        self.merge_mode = merge_mode

        # Create forward and backward layers
        self.forward_layer = YQuence(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            return_sequences=return_sequences,
            **kwargs
        )

        self.backward_layer = YQuence(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            return_sequences=return_sequences,
            **kwargs
        )

        self.return_sequences = return_sequences
        self.initialized = False
        self.cache = {}

    def to(self, device_type: str) -> 'BiYQuence':
        """Move layer to specified device"""
        super().to(device_type)
        self.forward_layer.to(device_type)
        self.backward_layer.to(device_type)
        return self

    def get_expected_input_shape(self) -> tuple:
        """Get expected input shape with None for flexible dimensions"""
        return (None, None, self.forward_layer.config['input_size'])

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """Compute output shape based on merge mode"""
        if self.merge_mode == 'concat':
            output_size = self.forward_layer.config['output_size'] * 2
        else:
            output_size = self.forward_layer.config['output_size']

        if self.return_sequences:
            return (input_shape[0], input_shape[1], output_size)
        return (input_shape[0], output_size)

    def _prepare_input(self, input_data):
        """Prepare input with proper shape handling"""
        if input_data.ndim != 3:
            input_data = self.shape_handler.auto_reshape(input_data, target_ndim=3)
        return input_data

    def _reverse_sequences(self, x, mask: Optional[Union[np.ndarray, 'cp.ndarray']] = None):
        """
        Reverse sequences for backward pass while respecting masks.
        """
        xp = self.device.xp

        if mask is None:
            return xp.flip(x, axis=1)

        reversed_sequences = []
        for seq, m in zip(x, mask):
            valid_length = xp.sum(m)
            valid_seq = seq[:valid_length]
            reversed_valid = xp.flip(valid_seq, axis=0)
            padded = xp.zeros_like(seq)
            padded[:valid_length] = reversed_valid
            reversed_sequences.append(padded)

        return xp.stack(reversed_sequences)

    def forward(self, x: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, 'cp.ndarray']:
        """Forward pass running both directions and merging results."""
        x = self._prepare_input(x)
        x = self.device.to_device(x)

        # Process forward direction
        forward_output = self.forward_layer.forward(x)
        self.mask = self.forward_layer.get_mask()

        # Handle backward direction
        if self.mask is not None:
            reversed_input = self._reverse_sequences(x, self.mask)
        else:
            reversed_input = self._reverse_sequences(x)

        backward_output = self.backward_layer.forward(reversed_input)

        # Re-reverse backward output to align with forward output
        if self.mask is not None:
            backward_output = self._reverse_sequences(backward_output, self.mask)
        else:
            backward_output = self._reverse_sequences(backward_output)

        # Store for backward pass
        self.cache.update({
            'x': x,
            'forward_output': forward_output,
            'backward_output': backward_output
        })

        return self._merge_outputs(forward_output, backward_output)

    def _merge_outputs(self, forward_output, backward_output):
        """Merge forward and backward outputs according to merge mode."""
        xp = self.device.xp

        if self.merge_mode == 'concat':
            return xp.concatenate([forward_output, backward_output], axis=-1)
        elif self.merge_mode == 'sum':
            return forward_output + backward_output
        elif self.merge_mode == 'mul':
            return forward_output * backward_output
        else:  # 'ave'
            return (forward_output + backward_output) / 2

    def backward(self, output_gradient: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Backward pass computing gradients for both directions."""
        xp = self.device.xp
        output_gradient = self.device.to_device(output_gradient)

        # Split gradient according to merge mode
        if self.merge_mode == 'concat':
            split_size = output_gradient.shape[-1] // 2
            forward_gradient = output_gradient[..., :split_size]
            backward_gradient = output_gradient[..., split_size:]
        elif self.merge_mode in ['sum', 'ave']:
            scale = 0.5 if self.merge_mode == 'ave' else 1.0
            forward_gradient = output_gradient * scale
            backward_gradient = output_gradient * scale
        else:  # 'mul'
            forward_output = self.cache['forward_output']
            backward_output = self.cache['backward_output']
            forward_gradient = output_gradient * backward_output
            backward_gradient = output_gradient * forward_output

        # Backward pass for forward layer
        forward_input_gradient = self.forward_layer.backward(forward_gradient)

        # Reverse backward gradient before backward pass
        reversed_backward_gradient = self._reverse_sequences(backward_gradient, self.mask)
        backward_input_gradient = self.backward_layer.backward(reversed_backward_gradient)

        # Re-reverse backward input gradient
        backward_input_gradient = self._reverse_sequences(backward_input_gradient, self.mask)

        # Combine input gradients
        return forward_input_gradient + backward_input_gradient

    def get_trainable_params(self) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """Get trainable parameters from both directions"""
        params = {}
        for prefix, layer in [('forward_', self.forward_layer),
                              ('backward_', self.backward_layer)]:
            layer_params = layer.get_trainable_params()
            for name, param in layer_params.items():
                params[prefix + name] = param
        return params

    def get_gradients(self) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """Get gradients from both directions"""
        grads = {}
        for prefix, layer in [('forward_', self.forward_layer),
                              ('backward_', self.backward_layer)]:
            layer_grads = layer.get_gradients()
            for name, grad in layer_grads.items():
                grads[prefix + name] = grad
        return grads

    def update_params(self, params: Dict[str, Union[np.ndarray, 'cp.ndarray']]):
        """Update parameters for both directions"""
        forward_params = {}
        backward_params = {}

        for name, param in params.items():
            if name.startswith('forward_'):
                forward_params[name[8:]] = param
            elif name.startswith('backward_'):
                backward_params[name[9:]] = param

        self.forward_layer.update_params(forward_params)
        self.backward_layer.update_params(backward_params)

    def get_config(self) -> dict:
        """Get layer configuration"""
        return {
            'merge_mode': self.merge_mode,
            'return_sequences': self.return_sequences,
            **self.forward_layer.get_config()
        }