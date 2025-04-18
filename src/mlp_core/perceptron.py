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
            # return lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)
            return lambda x: x
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
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the output of the perceptron given the inputs.
        
        Args:
            inputs: Input data
        
        Returns:
            the activation output of the perceptron
        """
        self.last_input = inputs
        z = np.dot(inputs, self.weights) + self.bias
        print("z.shape",z.shape)
        self.last_output = self.activation_fn(z)
        print(self.last_output.shape)
        return self.last_output
    
    def backward(self, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Perform the backward pass through the perceptron.
        
        Args:
            gradients: Gradient of the loss with respect to the perceptron's output
            learning_rate: Learning rate for weight updates
        
        Returns:
            Gradient of the loss with respect to the inputs
        """
        # Compute gradient w.r.t. weights and bias
        delta = gradients * self.activation_derivative(self.last_output)
        weight_gradients = np.dot(self.last_input.T, delta)
        bias_gradient = np.sum(delta, axis=0)
        
        # Update weights and bias
        self.weights -= learning_rate * weight_gradients
        self.bias -= learning_rate * bias_gradient
        
        # Compute gradient w.r.t. inputs for the previous layer
        input_gradients = np.dot(delta, self.weights.T)
        
        return input_gradients