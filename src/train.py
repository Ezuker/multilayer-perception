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
    parser.add_argument('--config', type=str, nargs='+', default='config/network.json',
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


def plot_training_history(networks):
    """Plot training and validation loss for multiple networks."""
    nb_networks = len(networks)
    n_cols = len(networks)  # Max 4 colonnes
    n_rows = (nb_networks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = np.array(axes).reshape(-1) if nb_networks > 1 else [axes]
    
    for i, (_, history) in enumerate(networks):
        if i < len(axes):
            ax = axes[i]
            if 'loss' in history:
                ax.plot(history['loss'], 'b-', label='Train')
            if 'val_loss' in history:
                ax.plot(history['val_loss'], 'r-', label='Val')
            ax.set_title(networks[i][0].name)
            ax.legend()
    
    # Cacher les axes inutilisés
    for i in range(nb_networks, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_comparison.png')
    plt.show()


def main():
    """Main function."""
    args = parse_args()
    
    if args.verbose:
        print(f"Loading configuration from: {args.config}")
    
    try:
        networks = []
        for model in args.config:
            layers_config, training_config = ConfigParser.parse_config(model)
            
            if args.verbose:
                print(f"Network architecture:")
                for i, layer in enumerate(layers_config):
                    if layer['type'] == 'dropout':
                        print(f"  Layer {i}: {layer['type']} - {layer['dropout']} dropout")
                    else:
                        print(f"  Layer {i}: {layer['type']} - {layer['units']} units, "
                            f"{layer['activation']} activation")
                print(f"Training parameters: {training_config}")
            
            network = Network(layers_config, training_config)
            print("Network initialized successfully!")
            if args.verbose:
                print(f"Network architecture: {network}")
                print(f"Ready to train using data from: {args.data_train}")
                print(f"Model will be saved to: {args.save}")
            
            x_train, y_train, x_val, y_val = ProcessData.get_data(args.data_train, args.data_validation)
            
            accumulated_history = {'loss': [], 'val_loss': []}
            history, best_network = network.fit(x_train, y_train, (x_val, y_val))
            
            accumulated_history['loss'].extend(history['loss'])
            if 'val_loss' in history and history['val_loss']:
                    accumulated_history['val_loss'].extend(history['val_loss'])

            print("Plotting accumulated training history...")
            print("Saving model...")
            best_network.save(f"{model}", history)
            best_network.name = model
            networks.append((best_network, history))
        plot_training_history(networks)
        

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()