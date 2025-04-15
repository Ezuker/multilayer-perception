from .perceptron import Perceptron

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
        
    def forward(self, inputs):
        """Compute the output of the layer given the inputs."""
        outputs = [p.forward(inputs) for p in self.perceptrons]
        return outputs
        
    def backward(self, gradients, learning_rate):
        pass