from .layer import Layer
import numpy as np
from tqdm import tqdm

class Network:
    """Complete neural network composed of multiple layers."""
    
    def __init__(self, layers_config: list[dict], training_config: dict):
        self.layers = []
        for i in range(1, len(layers_config)):
            input_size = layers_config[i-1]['units']
            output_size = layers_config[i]['units']
            activation = layers_config[i]['activation']
            weights_initializer = layers_config[i].get('weights_initializer', 'heUniform')
            
            layer = Layer(input_size, output_size, activation, weights_initializer)
            self.layers.append(layer)
        
        loss_name = training_config.get('loss', '')
        if loss_name != '':
            self.loss_function, self.loss_derivative = self._get_loss_function(loss_name)
        self.learning_rate = training_config.get('learning_rate', 0.01)
        self.batch_size = training_config.get('batch_size', 32)
        self.epochs = training_config.get('epochs', 100)

    def __str__(self):
        s = f"Network with {len(self.layers)} layers:\n"
        for i, layer in enumerate(self.layers):
            s += f"  Layer {i}: {layer}\n"
        return s.strip()
    
    @staticmethod
    def _get_loss_function(name: str):
        """Get the loss function and its derivative based on name."""
        if name == 'mean_squared_error':
            loss = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
            derivative = lambda y_true, y_pred: 2 * (y_pred - y_true) / y_true.shape[0]
            return loss, derivative
        elif name == 'binary_crossentropy':
            loss = lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-15) + 
                                                   (1 - y_true) * np.log(1 - y_pred + 1e-15))
            derivative = lambda y_true, y_pred: ((1 - y_true) / (1 - y_pred + 1e-15) - y_true / (y_pred + 1e-15)) / y_true.shape[0]
            return loss, derivative
        elif name == 'categoricalCrossentropy':
            # Note: Assumes y_pred is output of softmax
            loss = lambda y_true, y_pred: -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
            # The derivative of categorical cross-entropy combined with softmax is simply (y_pred - y_true)
            derivative = lambda y_true, y_pred: (y_pred - y_true) / y_true.shape[0]
            return loss, derivative
        raise ValueError(f"Unknown loss function: {name}")
        
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
        
    def backward(self, initial_gradient, learning_rate):
        """
        Perform the backward pass through the network using the initial loss gradient.
        
        Parameters:
        -----------
        initial_gradient : array-like
            The gradient of the loss function with respect to the network's output (dL/dy_pred).
        learning_rate : float
            Learning rate for weight updates
            
        Returns:
        --------
        None (updates layer weights internally)
        """
        gradient = initial_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
        
    def fit(self, X_train, y_train, validation_data=None):
        """
        Train the neural network on the provided data.
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training input data
        y_train : array-like, shape (n_samples, n_classes)
            Target values (one-hot encoded for classification)
        validation_data : tuple, optional
            Tuple of (X_val, y_val) for validation
            
        Returns:
        --------
        history : dict
            Dictionary containing training metrics (loss, val_loss) per epoch
        """
        n_samples = len(X_train)
        n_batches = max(1, n_samples // self.batch_size)
        history = {'loss': [], 'val_loss': []}
        
        for epoch in tqdm(range(self.epochs)):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Calculate loss
                batch_loss = self.loss_function(y_batch, y_pred)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
                
                # Calculate initial gradient for backpropagation
                initial_gradient = self.loss_derivative(y_batch, y_pred)
                
                # Backward pass
                self.backward(initial_gradient, self.learning_rate)
            
            history['loss'].append(epoch_loss)
            
            # Validation step
            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_function(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
                
            # Print progress
            if (epoch % 10 == 0 or epoch == self.epochs - 1):
                log_message = f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}"
                if val_loss is not None:
                    log_message += f", Val Loss: {val_loss:.4f}"
                print(log_message)
        
        return history
        
    def predict(self, X):
        """Generate predictions for input data X."""
        return self.forward(X)
    
    def save(self, filepath):
        """Save the model to a file."""
        import json
        
        # We can't directly save the layers due to unpicklable lambda functions
        # Instead, save the layer configurations and weights as JSON
        layers_data = []
        for layer in self.layers:
            # Convert numpy arrays to lists to make them JSON serializable
            weights = [p.weights.tolist() for p in layer.perceptrons]
            biases = [float(p.bias) for p in layer.perceptrons]
            
            layer_data = {
                'input_size': layer.config['input_size'],
                'output_size': layer.config['output_size'],
                'activation': layer.config['activation'],
                'weights_initializer': layer.config['weights_initializer'],
                'weights': weights,
                'biases': biases
            }
            layers_data.append(layer_data)
            
        # Save the network configuration and layer data
        network_data = {
            'layers_data': layers_data,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            # Also store the loss function name
            'loss': next((name for name, (func, _) in {
                'mean_squared_error': self._get_loss_function('mean_squared_error'),
                'binary_crossentropy': self._get_loss_function('binary_crossentropy'),
                'categoricalCrossentropy': self._get_loss_function('categoricalCrossentropy')
            }.items() if func == self.loss_function), '')
        }
        
        # Save as JSON file which can handle the data properly
        with open(filepath, 'w') as f:
            json.dump(network_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Load a model from a file."""
        import json
        from .layer import Layer
        
        with open(filepath, 'r') as f:
            network_data = json.load(f)
        
        # Create a minimal network structure to populate
        layers_config = []
        for layer_data in network_data['layers_data']:
            # Add input layer config
            if len(layers_config) == 0:
                layers_config.append({'units': layer_data['input_size']})
            # Add current layer config
            layers_config.append({
                'units': layer_data['output_size'],
                'activation': layer_data['activation'],
                'weights_initializer': layer_data['weights_initializer']
            })
        
        # Create training config
        training_config = {
            'learning_rate': network_data['learning_rate'],
            'batch_size': network_data['batch_size'],
            'epochs': network_data['epochs'],
            'loss': network_data.get('loss', '')  # Get loss name from saved data if available
        }
        
        # Create the network
        network = cls(layers_config, training_config)
        
        # Now load the weights and biases
        for i, layer_data in enumerate(network_data['layers_data']):
            for j, (weights, bias) in enumerate(zip(layer_data['weights'], layer_data['biases'])):
                network.layers[i].perceptrons[j].weights = np.array(weights)  # Convert list back to numpy array
                network.layers[i].perceptrons[j].bias = bias
                
        return network


