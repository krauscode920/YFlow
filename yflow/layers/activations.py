import numpy as np
from typing import Optional, Union
from ..core.layer import Layer


class Activation(Layer):
    """Base class for activation layers"""
    def __init__(self):
        super().__init__()

    def _forward_impl(self, x):
        raise NotImplementedError

    def _backward_impl(self, grad):
        raise NotImplementedError

    def forward(self, x):
        """Forward pass with GPU support"""
        x = self.device.to_device(x)
        self.input = x
        return self._forward_impl(x)

    def backward(self, grad):
        """Backward pass with GPU support"""
        grad = self.device.to_device(grad)
        return self._backward_impl(grad)


class ReLU(Activation):
    """ReLU activation with GPU support"""
    def _forward_impl(self, x):
        return self.device.xp.maximum(0, x)

    def _backward_impl(self, grad):
        return grad * (self.input > 0)


class LeakyReLU(Activation):
    """Leaky ReLU activation with GPU support"""
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def _forward_impl(self, x):
        xp = self.device.xp
        return xp.where(x > 0, x, x * self.alpha)

    def _backward_impl(self, grad):
        xp = self.device.xp
        return xp.where(self.input > 0, grad, grad * self.alpha)


class Sigmoid(Activation):
    """Sigmoid activation with GPU support"""
    def _forward_impl(self, x):
        xp = self.device.xp
        return 1 / (1 + xp.exp(-xp.clip(x, -709, 709)))

    def _backward_impl(self, grad):
        output = self._forward_impl(self.input)
        return grad * output * (1 - output)


class Tanh(Activation):
    """Tanh activation with GPU support"""
    def _forward_impl(self, x):
        return self.device.xp.tanh(x)

    def _backward_impl(self, grad):
        return grad * (1 - self.device.xp.square(self._forward_impl(self.input)))


class Softmax(Activation):
    """Softmax activation with GPU support"""
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def _forward_impl(self, x):
        xp = self.device.xp
        x_max = xp.max(x, axis=self.axis, keepdims=True)
        exp_x = xp.exp(x - x_max)
        return exp_x / xp.sum(exp_x, axis=self.axis, keepdims=True)

    def _backward_impl(self, grad):
        output = self._forward_impl(self.input)
        return output * (grad - (grad * output).sum(axis=self.axis, keepdims=True))


class ELU(Activation):
    """ELU activation with GPU support"""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def _forward_impl(self, x):
        xp = self.device.xp
        return xp.where(x > 0, x, self.alpha * (xp.exp(x) - 1))

    def _backward_impl(self, grad):
        xp = self.device.xp
        return xp.where(self.input > 0, grad, grad * self.alpha * xp.exp(self.input))


class SELU(Activation):
    """SELU activation with GPU support"""
    def __init__(self):
        super().__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def _forward_impl(self, x):
        xp = self.device.xp
        return self.scale * xp.where(x > 0, x, self.alpha * (xp.exp(x) - 1))

    def _backward_impl(self, grad):
        xp = self.device.xp
        return self.scale * xp.where(self.input > 0,
                                   grad,
                                   grad * self.alpha * xp.exp(self.input))