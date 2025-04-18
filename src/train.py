#!/usr/bin/env python3
# filepath: /home/bcarolle/multilayer-perception/src/train.py

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to allow imports from mlp_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlp_core.parser import ConfigParser
from mlp_core.network import Network
from data_tools.process_data_train import ProcessData

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Multilayer Perceptron Neural Network')
    parser.add_argument('--config', type=str, default='config/network.json',
                        help='Path to network configuration file')
    parser.add_argument('--data-train', type=str, required=True,
                        help='Path to training train data file')
    parser.add_argument('--data-validation', type=str, required=True,
                        help='Path to training validation data file')
    parser.add_argument('--save', type=str, default='models/model.pkl',
                        help='Path to save trained model')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()


def plot_training_history(history):
    """
    Plot training and validation loss from a training history dictionary.
    
    Args:
        history: Dictionary containing 'loss' and 'val_loss' lists
    """
    epochs = range(1, len(history['loss']) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot validation loss if available
    if 'val_loss' in history and history['val_loss']:
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No validation data', 
                 horizontalalignment='center', verticalalignment='center')
    
    # Improve layout and show plot
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def main():
    """Main function."""
    args = parse_args()
    
    if args.verbose:
        print(f"Loading configuration from: {args.config}")
    
    # try:
        layers_config, training_config = ConfigParser.parse_config(args.config)
        
        if args.verbose:
            print(f"Network architecture:")
            for i, layer in enumerate(layers_config):
                print(f"  Layer {i}: {layer['type']} - {layer['units']} units, "
                      f"{layer['activation']} activation")
            print(f"Training parameters: {training_config}")
        
        network = Network(layers_config, training_config)
        print("Network initialized successfully!")
        if args.verbose:
            print(f"Network architecture: {network}")
            print(f"Ready to train using data from: {args.data_train}")
            print(f"Model will be saved to: {args.save}")
        
        x_train, y_train, x_val, y_val, x_mean, x_std = ProcessData.get_data(args.data_train, args.data_validation)
        x_val = (x_val - x_mean) / x_std
        history = network.fit(x_train, y_train, (x_val, y_val))
        plot_training_history(history)
        
    # except Exception as e:
    #     print(f"Error: {e}")
    #     sys.exit(1)

if __name__ == '__main__':
    main()