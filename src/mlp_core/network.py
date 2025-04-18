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
        self.loss_function = self._get_loss_function(training_config.get('loss', ''))
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
        """Get the loss function based on name."""
        if name == 'mean_squared_error':
            return lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
        elif name == 'binary_crossentropy':
            return lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-15) + 
                                                   (1 - y_true) * np.log(1 - y_pred + 1e-15))
        elif name == 'categoricalCrossentropy':
            return lambda y_true, y_pred: -np.sum(y_true * np.log(y_pred + 1e-15), axis=1)
        raise ValueError(f"Unknown loss function: {name}")
        
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
        
    def backward(self, y_true, y_pred, learning_rate):
        """
        Perform the backward pass through the network.
        
        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_classes)
            True labels (one-hot encoded for classification)
        y_pred : array-like, shape (n_samples, n_classes)
            Predicted labels
        learning_rate : float
            Learning rate for weight updates
            
        Returns:
        --------
        gradients : array-like, shape (n_samples, n_features)
            Gradients of the loss with respect to the inputs
        """
        loss_gradient = self.loss_function(y_true, y_pred)
        
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)
        
        return loss_gradient
        
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
                
                y_pred = self.forward(X_batch)
                batch_loss = self.loss_function(y_batch, y_pred)
                if isinstance(batch_loss, np.ndarray):
                    batch_loss = np.mean(batch_loss)
                
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
                
                self.backward(y_batch, y_pred, self.learning_rate)
            
            history['loss'].append(epoch_loss)
            
            # Validation step
            # if validation_data is not None:
            #     X_val, y_val = validation_data
            #     y_val_pred = self.forward(X_val)
            #     val_loss = self.loss_function(y_val, y_val_pred)
            #     if isinstance(val_loss, np.ndarray):
            #         val_loss = np.mean(val_loss)
            #     history['val_loss'].append(val_loss)
                
            #     if epoch % 10 == 0:
            #         print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            if (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")
        
        return history
        
    def predict(self, X):
        pass