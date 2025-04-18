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
        
    def __str__(self):
        return f"Layer with {len(self.perceptrons)} perceptrons, {self.config}"
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the output of the layer given the inputs.
        """
        if self.config['activation'] == 'softmax':
            logits = np.zeros((inputs.shape[0], len(self.perceptrons)))
            for i, p in enumerate(self.perceptrons):
                p.last_input = inputs
                logits[:, i] = np.dot(inputs, p.weights) + p.bias
            
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            softmax_output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            for i, p in enumerate(self.perceptrons):
                p.last_output = softmax_output[:, i:i+1]
            
            print("Layer forward pass output shape:", softmax_output.shape)
            return softmax_output        
        else:
            print("Layer forward pass input shape:", inputs.shape)
            outputs = [p.forward(inputs) for p in self.perceptrons]
            print("Layer forward pass output shape:", np.array(outputs).T.shape)
            return np.array(outputs).T

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