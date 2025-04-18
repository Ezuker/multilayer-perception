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
        # print("Logits shape:", logits.shape)
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
        # print("Layer forward pass output shape:", outputs.shape)
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
        else:
            # For other activations, compute gradients through activation function
            if len(gradients.shape) == 1 or gradients.shape[1] == 1:
                # Handle 1D gradients case
                activation_gradients = np.zeros((batch_size, len(self.perceptrons)))
                for i, p in enumerate(self.perceptrons):
                    if self.config['activation'] == 'sigmoid':
                        activation_gradients[:, i] = gradients.flatten() * p.last_activation.flatten() * (1 - p.last_activation.flatten())
                    elif self.config['activation'] == 'relu':
                        activation_gradients[:, i] = gradients.flatten() * (p.last_output.flatten() > 0)
                    elif self.config['activation'] == 'tanh':
                        activation_gradients[:, i] = gradients.flatten() * (1 - p.last_activation.flatten()**2)
            else:
                # Original code for 2D gradients
                activation_gradients = np.zeros_like(gradients)
                for i, p in enumerate(self.perceptrons):
                    if self.config['activation'] == 'sigmoid':
                        activation_gradients[:, i:i+1] = gradients[:, i:i+1] * p.last_activation * (1 - p.last_activation)
                    elif self.config['activation'] == 'relu':
                        activation_gradients[:, i:i+1] = gradients[:, i:i+1] * (p.last_output > 0)
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