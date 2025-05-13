"""
Optimizers for neural network training.

This module implements various optimization algorithms for training neural networks,
including SGD with momentum, RMSprop, and Adam.
"""

import numpy as np


class Optimizer:
    """Base class for optimizers."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, weights, gradients):
        """
        Update weights based on gradients.
        
        Parameters:
        -----------
        weights : array-like
            Current weights
        gradients : array-like
            Gradients of the weights
            
        Returns:
        --------
        array-like
            Updated weights
        """
        raise NotImplementedError("Subclasses must implement this method")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Basic implementation of SGD without momentum.
    """
    
    def update(self, weights, gradients):
        """Update weights using SGD."""
        return weights - self.learning_rate * gradients


class SGDMomentum(Optimizer):
    """
    SGD with Momentum optimizer.
    
    Implements momentum to accelerate gradient descent and reduce oscillations.
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        
    def update(self, weights, gradients):
        """Update weights using SGD with momentum."""
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)
            
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        return weights + self.velocity


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Maintains a moving average of squared gradients to normalize the gradients.
    This helps with training stability by adapting the learning rate for each parameter.
    """
    
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.square_avg = None
        
    def update(self, weights, gradients):
        """Update weights using RMSprop."""
        if self.square_avg is None:
            self.square_avg = np.zeros_like(weights)
            
        self.square_avg = self.decay_rate * self.square_avg + (1 - self.decay_rate) * gradients**2
        return weights - self.learning_rate * gradients / (np.sqrt(self.square_avg) + self.epsilon)


class Adam(Optimizer):
    """
    Adam optimizer.
    
    Adaptive Moment Estimation combines ideas from both momentum and RMSprop.
    Maintains both a moving average of gradients (momentum) and a moving average of 
    squared gradients (RMSprop).
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1  # Exponential decay rate for the first moment estimates
        self.beta2 = beta2  # Exponential decay rate for the second moment estimates
        self.epsilon = epsilon
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0  # Timestep
        
    def update(self, weights, gradients):
        """Update weights using Adam."""
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)
            
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def get_optimizer(name, **kwargs):
    """
    Factory function to get an optimizer instance by name.
    
    Parameters:
    -----------
    name : str
        Name of the optimizer ('sgd', 'sgd_momentum', 'rmsprop', 'adam')
    **kwargs : dict
        Additional parameters for the optimizer
        
    Returns:
    --------
    Optimizer
        Instance of the requested optimizer
    """
    optimizers = {
        'sgd': SGD,
        'sgd_momentum': SGDMomentum,
        'rmsprop': RMSprop,
        'adam': Adam
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available optimizers: {list(optimizers.keys())}")
    
    return optimizers[name.lower()](**kwargs)
