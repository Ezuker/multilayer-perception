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
        self.last_output = z
        # print("z.shape",z.shape)
        return self.last_output
    
    def backward(self, gradients, learning_rate):
        """
        Update weights and compute gradients for inputs.
        """        
        # Compute weight gradients (should match weights shape)
        weight_gradients = np.dot(self.last_input.T, gradients)
        
        # Flatten if needed to match weights shape
        if len(weight_gradients.shape) > 1:
            weight_gradients = np.sum(weight_gradients, axis=1)
        
        bias_gradient = np.sum(gradients, axis=0)
        
        # Update weights and bias
        self.weights -= learning_rate * weight_gradients
        self.bias -= learning_rate * bias_gradient
        
        # Compute gradients for inputs (for previous layer)
        input_gradients = np.dot(gradients, self.weights.reshape(1, -1))
        
        return input_gradients