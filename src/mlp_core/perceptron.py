import numpy as np
from typing import Callable
from .optimizers import SGD

class Perceptron:
    """
    A single neuron (perceptron) implementation.
    
    This class represents one unit in a neural network layer, managing
    its own weights, bias, and activation function.
    """
    
    def __init__(self, input_size: int, 
                 activation: str = 'sigmoid',
                 weights_initializer: str = 'heUniform',
                 optimizer=None):
        """
        Initialize a perceptron with given input size and activation.
        
        Args:
            input_size: Number of input features
            activation: Activation function name ('sigmoid', 'relu', etc.)
            weights_initializer: Method to initialize weights
            optimizer: Optimizer instance to use for weight updates
        """
        self.input_size = input_size
        self.activation_name = activation
        
        self.weights = self._initialize_weights(input_size, weights_initializer)
        self.bias = 0.0
            
        self.last_X = None
        self.last_Z = None
        
        # Default optimizer is SGD if none provided
        self.optimizer = optimizer if optimizer is not None else SGD()
    
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
        self.last_X = inputs
        z = np.dot(inputs, self.weights) + self.bias
        self.last_Z = z
        return self.last_Z
    
    def backward(self, gradients, learning_rate):
        """
        Update weights and compute gradients for inputs.
        
        Args:
            gradients: Gradients from the next layer
            learning_rate: Learning rate for weight updates
            
        Returns:
            Gradients to propagate to the previous layer
        """        
        # Compute weight gradients (should match weights shape)
        weight_gradients = np.dot(self.last_X.T, gradients)
        
        # Flatten if needed to match weights shape
        if len(weight_gradients.shape) > 1:
            weight_gradients = np.sum(weight_gradients, axis=1)
        
        bias_gradient = np.sum(gradients, axis=0)
        
        # Set optimizer learning rate
        self.optimizer.learning_rate = learning_rate
        
        # Update weights using optimizer
        self.weights = self.optimizer.update(self.weights, weight_gradients)
        self.bias -= learning_rate * bias_gradient  # Simple update for bias
        
        # Compute gradients for inputs (for previous layer)
        input_gradients = np.dot(gradients, self.weights.reshape(1, -1))
        
        return input_gradients