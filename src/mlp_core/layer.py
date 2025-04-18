from .perceptron import Perceptron
import numpy as np

class Layer:
    """Collection of perceptrons forming one layer in the neural network."""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: str = 'sigmoid',
                 weights_initializer: str = 'heUniform'):
        self.config = {
            'input_size': input_size,
            'output_size': output_size,
            'activation': activation,
            'weights_initializer': weights_initializer
        }
        self.perceptrons = [Perceptron(input_size, activation, weights_initializer) 
                             for _ in range(output_size)]
        
    @staticmethod
    def _get_activation_function(name: str):
        """Get the activation function based on name."""
        if name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        elif name == 'softmax':
            return lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        raise ValueError(f"Unknown activation function: {name}")
    
    def __str__(self):
        return f"Layer with {len(self.perceptrons)} perceptrons, {self.config}"
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the output of the layer given the inputs.
        """
        logits = np.array([p.forward(inputs) for p in self.perceptrons]).T
        print("Logits shape:", logits.shape)
        if self.config['activation'] == 'sigmoid':
            outputs = 1 / (1 + np.exp(-logits))
        elif self.config['activation'] == 'relu':
            outputs = np.maximum(0, logits)
        elif self.config['activation'] == 'tanh':
            outputs = np.tanh(logits)
        elif self.config['activation'] == 'softmax':
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            outputs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {self.config['activation']}")
        
        for i, p in enumerate(self.perceptrons):
            p.last_activation = outputs[:, i:i+1]
        print("Layer forward pass output shape:", outputs.shape)
        return outputs

    def backward(self, gradients, learning_rate):
        """
        Perform the backward pass through the layer.
        
        Args:
            gradients: Gradient of the loss with respect to the layer's outputs
            learning_rate: Learning rate for weight updates
            
        Returns:
            Gradient of the loss with respect to the layer's inputs (for previous layer)
        """
        batch_size = gradients.shape[0]
        input_gradients = np.zeros((batch_size, self.config['input_size']))
        
        # For each perceptron in the layer
        for i, perceptron in enumerate(self.perceptrons):
            # Pass the corresponding gradient for this perceptron
            perceptron_gradients = gradients[:, i].reshape(-1, 1)
            # The backward pass through each perceptron returns gradients w.r.t inputs
            input_grad_i = perceptron.backward(perceptron_gradients, learning_rate)
            # Accumulate gradients from all perceptrons
            input_gradients += input_grad_i
        print("Layer backward pass input gradients:", input_gradients.shape)
        return input_gradients