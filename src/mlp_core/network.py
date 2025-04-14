from .layer import Layer

class Network:
    """Complete neural network composed of multiple layers."""
    
    def __init__(self, layers_config: list[dict]):
        self.layers = []
        for i in range(1, len(layers_config)):
            input_size = layers_config[i-1]['units']
            output_size = layers_config[i]['units']
            activation = layers_config[i]['activation']
            weights_initializer = layers_config[i].get('weights_initializer', 'heUniform')
            
            layer = Layer(input_size, output_size, activation, weights_initializer)
            self.layers.append(layer)

    def __str__(self):
        s = f"Network with {len(self.layers)} layers:\n"
        for i, layer in enumerate(self.layers):
            s += f"  Layer {i}: {layer}\n"
        return s.strip()
        
    def forward(self, inputs):
        pass
        
    def backward(self, y_true, y_pred, learning_rate):
        pass
        
    def fit(self, X_train, y_train, **training_params):
        pass
        
    def predict(self, X):
        pass