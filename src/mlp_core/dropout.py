import numpy as np

class Dropout:
    """
    Dropout layer for regularization.
    
    During training, randomly sets a fraction of inputs to zero.
    During inference, scales outputs by the keep probability.
    """
    
    def __init__(self, keep_prob=0.5, units=None):
        """
        Initialize dropout layer.
        
        Args:
            keep_prob: Probability of keeping a neuron active (1-dropout_rate)
        """
        self.keep_prob = keep_prob
        self.mask = None
        self.training_mode = True
        self.units = units
    
    def forward(self, inputs):
        """
        Apply dropout during forward pass.
        
        Args:
            inputs: Input data
            
        Returns:
            Outputs with dropout applied during training
        """
        if self.training_mode:
            self.mask = np.random.binomial(1, self.keep_prob, size=inputs.shape) / self.keep_prob
            return inputs * self.mask
        else:
            return inputs
    
    def backward(self, gradients, learning_rate=None):
        """
        Propagate gradients through dropout layer.
        
        Args:
            gradients: Gradients from next layer
            learning_rate: Not used in dropout (no parameters to update)
            
        Returns:
            Gradients for previous layer
        """
        if self.training_mode:
            return gradients * self.mask
        else:
            return gradients
    
    def set_training(self, mode=True):
        """Set layer to training or inference mode."""
        self.training_mode = mode
        
    def __str__(self):
        """String representation of the dropout layer."""
        return f"Dropout layer (keep_prob={self.keep_prob})"
