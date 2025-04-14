import numpy as np
from typing import Callable

class Perceptron:
    """
    A single neuron (perceptron) implementation.
    
    This class represents one unit in a neural network layer, managing
    its own weights, bias, and activation function.
    """
    
    def __init__(self, input_size: int, 
                 activation: str = 'sigmoid',
                 weights_initializer: str = 'heUniform'):
        """
        Initialize a perceptron with given input size and activation.
        
        Args:
            input_size: Number of input features
            activation: Activation function name ('sigmoid', 'relu', etc.)
            weights_initializer: Method to initialize weights
        """
        self.input_size = input_size
        self.activation_name = activation
        
        self.weights = self._initialize_weights(input_size, weights_initializer)
        self.bias = 0.0
        
        self.activation_fn = self._get_activation_function(activation)
        self.activation_derivative = self._get_activation_derivative(activation)
        
        self.last_input = None
        self.last_output = None
        self.last_activation = None
    
    def _initialize_weights(self, size: int, method: str) -> np.ndarray:
        """Initialize weights based on chosen method."""
        if method == 'heUniform':
            scale = np.sqrt(2.0 / size)
            return np.random.randn(size) * scale
        elif method == 'xavierUniform':
            scale = np.sqrt(1.0 / size)
            return np.random.randn(size) * scale
        else:
            return np.random.randn(size) * 0.01
    
    def _get_activation_function(self, name: str) -> Callable:
        """Get the activation function based on name."""
        if name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        elif name == 'softmax':
            return lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)
        raise ValueError(f"Unknown activation function: {name}")
    
    def _get_activation_derivative(self, name: str) -> Callable:
        """Get the derivative of the activation function."""
        if name == 'sigmoid':
            return lambda x: x * (1 - x)
        elif name == 'relu':
            return lambda x: np.where(x > 0, 1, 0)
        elif name == 'tanh':
            return lambda x: 1 - x**2
        elif name == 'softmax':
            def softmax_derivative(x):
                s = x.reshape(-1, 1)
                return np.diagflat(s) - np.dot(s, s.T)
            return softmax_derivative
        else:
            raise ValueError(f"Unknown activation function: {name}")