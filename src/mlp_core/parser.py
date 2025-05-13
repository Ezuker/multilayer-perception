import json
import os
from typing import Dict, List, Any, Tuple

class ConfigParser:
    """
    Parser for neural network configuration files in JSON format.
    Handles loading, validating, and extracting network architecture and training parameters.
    """

    @staticmethod
    def parse_config(config_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parse the network configuration file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            Tuple containing (layers_config, training_config)
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config file is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in configuration file: {config_path}")
        
        ConfigParser._validate_config_structure(config)
        layers_config = config['network']['layers']
        training_config = config['training']
        ConfigParser._validate_layers_config(layers_config)
        ConfigParser._validate_training_config(training_config)
        
        return layers_config, training_config
    
    @staticmethod
    def _validate_config_structure(config: Dict) -> None:
        """Validate the overall structure of the config."""
        if 'network' not in config:
            raise ValueError("Missing 'network' section in configuration")
        
        if 'layers' not in config['network']:
            raise ValueError("Missing 'layers' section in network configuration")
        
        if 'training' not in config:
            raise ValueError("Missing 'training' section in configuration")
        
        if not isinstance(config['network']['layers'], list):
            raise ValueError("'layers' must be a list")
        
        if not config['network']['layers']:
            raise ValueError("Network must have at least one layer")
    
    @staticmethod
    def _validate_layers_config(layers: List[Dict]) -> None:
        """Validate each layer in the configuration."""
        if layers[0]['type'] != 'input':
            raise ValueError("First layer must be of type 'input'")
        
        if layers[-1]['type'] != 'output':
            raise ValueError("Last layer must be of type 'output'")
        
        hidden_layers = [layer for layer in layers if layer['type'] == 'hidden']
        if len(hidden_layers) < 2:
            raise ValueError("Network must have at least two hidden layers")
        
        for i, layer in enumerate(layers):
            if 'type' not in layer:
                raise ValueError(f"Layer {i} is missing 'type' attribute")
            
            if 'units' not in layer:
                raise ValueError(f"Layer {i} is missing 'units' attribute")
            
            if 'activation' not in layer:
                raise ValueError(f"Layer {i} is missing 'activation' attribute")
            
            valid_activations = ['sigmoid', 'relu', 'tanh', 'softmax', 'linear', 'leakyRelu']
            if layer['activation'] not in valid_activations:
                raise ValueError(f"Layer {i} has invalid activation function: {layer['activation']}")
            
            valid_types = ['input', 'hidden', 'output']
            if layer['type'] not in valid_types:
                raise ValueError(f"Layer {i} has invalid type: {layer['type']}")
            
            if i > 0 and 'weights_initializer' not in layer:
                layer['weights_initializer'] = 'heUniform'
            
            if i > 0 and i < len(layers) - 1 and layer['type'] != 'hidden':
                raise ValueError(f"Layer {i} must be of type 'hidden' if not input or output")
    
    @staticmethod
    def _validate_training_config(training: Dict) -> None:
        """Validate the training configuration."""
        required_fields = ['loss', 'batch_size', 'epochs']
        for field in required_fields:
            if field not in training:
                raise ValueError(f"Training configuration is missing '{field}'")
        
        # Check if learning_rate is present directly or within optimizer_params
        has_learning_rate = False
        if 'learning_rate' in training:
            has_learning_rate = True
            if not isinstance(training['learning_rate'], (int, float)) or training['learning_rate'] <= 0:
                raise ValueError(f"Invalid learning rate: {training['learning_rate']}")
        elif 'optimizer_params' in training and 'learning_rate' in training['optimizer_params']:
            has_learning_rate = True
            if not isinstance(training['optimizer_params']['learning_rate'], (int, float)) or training['optimizer_params']['learning_rate'] <= 0:
                raise ValueError(f"Invalid learning rate: {training['optimizer_params']['learning_rate']}")
        
        if not has_learning_rate:
            raise ValueError("Training configuration is missing 'learning_rate' either directly or in 'optimizer_params'")
        
        if not isinstance(training['batch_size'], int) or training['batch_size'] <= 0:
            raise ValueError(f"Invalid batch size: {training['batch_size']}")
        
        if not isinstance(training['epochs'], int) or training['epochs'] <= 0:
            raise ValueError(f"Invalid epochs: {training['epochs']}")
        
        valid_losses = ['categoricalCrossentropy', 'mse', 'binaryCrossentropy']
        if training['loss'] not in valid_losses:
            raise ValueError(f"Invalid loss function: {training['loss']}")


def parse_network_config(config_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convenience function to parse network configuration."""
    return ConfigParser.parse_config(config_path)