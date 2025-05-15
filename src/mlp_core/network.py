from .layer import Layer
from .dropout import Dropout
import numpy as np
from tqdm import tqdm
from .optimizers import get_optimizer

class Network:
    """Complete neural network composed of multiple layers."""
    
    def __init__(self, layers_config: list[dict], training_config: dict):
        optimizer_name = training_config.get('optimizer', 'sgd')
        optimizer_params = training_config.get('optimizer_params', {})
        
        self.optimizer = get_optimizer(optimizer_name, **optimizer_params)
        
        self.layers = []
        print(layers_config)
        for i in range(1, len(layers_config)):
            if layers_config[i].get("type") == 'dropout':
                dropout_rate = layers_config[i]['dropout']
                units = layers_config[i]['units']
                self.layers.append(Dropout(keep_prob=1 - dropout_rate, units=units))
                continue
            input_size = layers_config[i-1]['units']
            output_size = layers_config[i]['units']
            activation = layers_config[i]['activation']
            weights_initializer = layers_config[i].get('weights_initializer', 'heUniform')
            
            layer = Layer(input_size, output_size, activation, weights_initializer, 
                         optimizer=get_optimizer(optimizer_name, **optimizer_params))
            self.layers.append(layer)
        
        loss_name = training_config.get('loss', '')
        if loss_name != '':
            self.loss_function, self.loss_derivative = self._get_loss_function(loss_name)
        self.learning_rate = training_config.get('learning_rate', 0.01)
        self.batch_size = training_config.get('batch_size', 32)
        self.epochs = training_config.get('epochs', 100)
        self.patience = training_config.get('patience', 10)
        self.min_delta = training_config.get('min_delta', 0.001)

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
            derivative = lambda y_true, y_pred: (y_pred - y_true) / y_true.shape[0]
            return loss, derivative
        elif name == 'categoricalCrossentropy':
            loss = lambda y_true, y_pred: -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
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
        def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
            y_pred = (y_pred_prob > threshold).astype(int)
            
            # Précision globale
            accuracy = np.mean(y_pred == y_true)
            
            # Calcul matrice de confusion
            true_pos = np.sum((y_true == 1) & (y_pred == 1))
            false_pos = np.sum((y_true == 0) & (y_pred == 1))
            true_neg = np.sum((y_true == 0) & (y_pred == 0))
            false_neg = np.sum((y_true == 1) & (y_pred == 0))
            
            # Métriques dérivées
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = true_pos / (true_pos + 0.5 * (false_pos + false_neg)) if (true_pos + 0.5 * (false_pos + false_neg)) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_pos': true_pos,
                'false_pos': false_pos,
                'true_neg': true_neg,
                'false_neg': false_neg
            }
        n_samples = len(X_train)
        n_batches = max(1, n_samples // self.batch_size)
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        best_val_loss = float('inf')
        best_network = None
        patience_counter = 0
        
        for epoch in range(self.epochs):
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
                batch_metrics = calculate_metrics(y_batch, y_pred)
                for metric_name, value in batch_metrics.items():
                    if f'train_{metric_name}' not in history:
                        history[f'train_{metric_name}'] = []
                    history[f'train_{metric_name}'].append(value)
                
                # Calculate loss
                batch_loss = self.loss_function(y_batch, y_pred)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
                
                # Calculate initial gradient for backpropagation
                initial_gradient = self.loss_derivative(y_batch, y_pred)
                
                # Backward pass
                self.backward(initial_gradient, self.learning_rate)

            
            history['loss'].append(epoch_loss)
            
            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_function(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
                batch_metrics = calculate_metrics(y_val, y_val_pred)
                for metric_name, value in batch_metrics.items():
                    if f'val_{metric_name}' not in history:
                        history[f'val_{metric_name}'] = []
                    history[f'val_{metric_name}'].append(value)
                
                if val_loss <= best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    best_network = self.clone()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                    
            if (epoch % 10 == 0 or epoch == self.epochs - 1):
                accuracy = history['train_accuracy'][-1] if 'train_accuracy' in history else None
                log_message = f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}"
                if val_loss is not None:
                    accuracy = history['val_accuracy'][-1] if 'val_accuracy' in history else None
                    log_message += f", Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}"
                print(log_message)
        
        print("Best validation loss:", best_val_loss, "at epoch", history['val_loss'].index(best_val_loss) + 1)
        return history, best_network
        
    def predict(self, X):
        """Generate predictions for input data X."""
        return self.forward(X)
    
    def save(self, filepath, history=None):
        """Save the model to a file."""
        import json
        
        layers_data = []
        for layer in self.layers:
            # Skip dropout layers because we don't need it while predicting
            if isinstance(layer, Dropout):
                continue
            weights = [p.weights.tolist() for p in layer.perceptrons]
            biases = [float(p.bias) for p in layer.perceptrons]
            
            layer_data = {
                'input_size': layer.config['input_size'],
                'output_size': layer.config['output_size'],
                'activation': layer.config['activation'],
                'weights_initializer': layer.config['weights_initializer'],
                'weights': weights,
                'biases': biases,
            }
            layers_data.append(layer_data)
            
        network_data = {
            'layers_data': layers_data,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'loss': next((name for name, (func, _) in {
                'mean_squared_error': self._get_loss_function('mean_squared_error'),
                'binary_crossentropy': self._get_loss_function('binary_crossentropy'),
                'categoricalCrossentropy': self._get_loss_function('categoricalCrossentropy')
            }.items() if func == self.loss_function), '')
        }
        
        with open(filepath, 'w') as f:
            json.dump(network_data, f, indent=2)
        if history:
            serializable_history = {}
            num_epochs = len(history['loss'])
            for epoch in range(num_epochs):
                serializable_history[str(epoch+1)] = {}
                
                for key, values in history.items():
                    if epoch < len(values):
                        value = values[epoch]
                        if isinstance(value, (np.integer, np.int64, np.int32)):
                            serializable_history[str(epoch+1)][key] = int(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32)):
                            serializable_history[str(epoch+1)][key] = float(value)
                        elif isinstance(value, np.ndarray):
                            serializable_history[str(epoch+1)][key] = value.tolist()
                        else:
                            serializable_history[str(epoch+1)][key] = value
            
            with open(filepath + '_history.json', 'w') as f:
                json.dump(serializable_history, f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Load a model from a file."""
        import json
        
        with open(filepath, 'r') as f:
            network_data = json.load(f)
        
        layers_config = []
        for layer_data in network_data['layers_data']:
            if len(layers_config) == 0:
                layers_config.append({'units': layer_data['input_size']})
            layers_config.append({
                'units': layer_data['output_size'],
                'activation': layer_data['activation'],
                'weights_initializer': layer_data['weights_initializer']
            })
        
        training_config = {
            'learning_rate': network_data['learning_rate'],
            'batch_size': network_data['batch_size'],
            'epochs': network_data['epochs'],
            'loss': network_data.get('loss', '')
        }
        
        network = cls(layers_config, training_config)
        
        for i, layer_data in enumerate(network_data['layers_data']):
            for j, (weights, bias) in enumerate(zip(layer_data['weights'], layer_data['biases'])):
                network.layers[i].perceptrons[j].weights = np.array(weights)  # Convert list back to numpy array
                network.layers[i].perceptrons[j].bias = bias
        return network
    
    def clone(self):
        """Create a deep copy of the network."""
        import copy
        return copy.deepcopy(self)