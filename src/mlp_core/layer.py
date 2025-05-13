from .perceptron import Perceptron
import numpy as np

class Layer:
    """Collection of perceptrons forming one layer in the neural network."""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: str = 'sigmoid',
                 weights_initializer: str = 'heUniform',
                 optimizer=None):
        self.config = {
            'input_size': input_size,
            'output_size': output_size,
            'activation': activation,
            'weights_initializer': weights_initializer
        }
        self.perceptrons = [Perceptron(input_size, activation, weights_initializer, optimizer) 
                             for _ in range(output_size)]
        self.activation_function = self._get_activation_function(activation)
        
    @staticmethod
    def _get_activation_function(name: str):
        """Get the activation function based on name."""
        if name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'leakyRelu':
            return lambda x: np.where(x > 0, x, 0.01 * x)  # alpha=0.01 is the standard leaky factor
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        elif name == 'linear':
            return lambda x: x  # Identity function that returns input unchanged
        elif name == 'softmax':
            def softmax(x):
                exp_logits = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return softmax
        raise ValueError(f"Unknown activation function: {name}")
    
    def __str__(self):
        return f"Layer with {len(self.perceptrons)} perceptrons, {self.config}"
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the output of the layer given the inputs.
        """
        Z_list = np.array([p.forward(inputs) for p in self.perceptrons]).T
        outputs = self.activation_function(Z_list)
        
        for i, p in enumerate(self.perceptrons):
            p.last_activation = outputs[:, i:i+1]
        return outputs

    def backward(self, gradients, learning_rate):
        """
        Perform the backward pass through the layer.
        """
        batch_size = gradients.shape[0]
        
        # Reshape gradients if they're 1D
        if len(gradients.shape) == 1:
            gradients = gradients.reshape(-1, 1)
        
        input_gradients = np.zeros((batch_size, self.config['input_size']))
        
        # Special case for softmax (when used with cross-entropy loss)
        if self.config['activation'] == 'softmax':
            activation_gradients = gradients
        elif self.config['activation'] == 'linear':
            # Derivative of linear/identity function is 1
            activation_gradients = gradients
        else:
            # For other activations, compute gradients through activation function
            if len(gradients.shape) == 1 or gradients.shape[1] == 1:
                # Handle 1D gradients case
                activation_gradients = np.zeros((batch_size, len(self.perceptrons)))
                for i, p in enumerate(self.perceptrons):
                    if self.config['activation'] == 'sigmoid':
                        activation_gradients[:, i] = gradients.flatten() * p.last_activation.flatten() * (1 - p.last_activation.flatten())
                    elif self.config['activation'] == 'relu':
                        # Reshape last_Z to ensure proper broadcasting
                        relu_mask = (p.last_Z.flatten() > 0)
                        activation_gradients[:, i] = gradients.flatten() * relu_mask
                    elif self.config['activation'] == 'leakyRelu':
                        # Derivative of leaky ReLU is 1 where x > 0, alpha otherwise
                        leaky_mask = np.where(p.last_Z.flatten() > 0, 1, 0.01)
                        activation_gradients[:, i] = gradients.flatten() * leaky_mask
                    elif self.config['activation'] == 'tanh':
                        activation_gradients[:, i] = gradients.flatten() * (1 - p.last_activation.flatten()**2)
            else:
                # Original code for 2D gradients
                activation_gradients = np.zeros_like(gradients)
                for i, p in enumerate(self.perceptrons):
                    if self.config['activation'] == 'sigmoid':
                        activation_gradients[:, i:i+1] = gradients[:, i:i+1] * p.last_activation * (1 - p.last_activation)
                    elif self.config['activation'] == 'relu':
                        # Shape p.last_Z to match the dimensions required for broadcasting
                        relu_mask = (p.last_Z.reshape(-1, 1) > 0)
                        activation_gradients[:, i:i+1] = gradients[:, i:i+1] * relu_mask
                    elif self.config['activation'] == 'leakyRelu':
                        # Derivative of leaky ReLU is 1 where x > 0, alpha otherwise
                        leaky_mask = np.where(p.last_Z.reshape(-1, 1) > 0, 1, 0.01)
                        activation_gradients[:, i:i+1] = gradients[:, i:i+1] * leaky_mask
                    elif self.config['activation'] == 'tanh':
                        activation_gradients[:, i:i+1] = gradients[:, i:i+1] * (1 - p.last_activation**2)
        
        # For each perceptron, compute weight updates and input gradients
        for i, perceptron in enumerate(self.perceptrons):
            if len(activation_gradients.shape) == 1:
                perceptron_gradients = activation_gradients.reshape(-1, 1)
            else:
                if activation_gradients.shape[1] == 1:
                    perceptron_gradients = activation_gradients
                else:
                    perceptron_gradients = activation_gradients[:, i:i+1]
            
            input_grad_i = perceptron.backward(perceptron_gradients, learning_rate)
            input_gradients += input_grad_i
        
        return input_gradients