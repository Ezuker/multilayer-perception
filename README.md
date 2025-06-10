# Multilayer Perceptron (MLP) Project

This project aims to build a **Multilayer Perceptron (MLP)** neural network from scratch using Python, focusing on understanding key machine learning concepts such as **feedforward**, **backpropagation**, and **gradient descent**.

## Features

- **Data Splitting & Visualization**:
  - Split dataset into training and validation sets.
  - Visualize dataset distributions and relationships.

- **MLP Model**:
  - Build a **Multilayer Perceptron** with configurable layers and activation functions.
  - Train the model using different optimizers: default, Adam, Momentum, and RMSprop.

- **Prediction**:
  - Use the trained models to make predictions on new data.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url> && cd multilayer-perception
   ```

2. **Setup the environment**:
   ```bash
   make setup
   ```
   This will:
   - Create a Python virtual environment
   - Install dependencies from requirements.txt
   - Create necessary directories for data and models

   Next, **activate the virtual environment** (run this in your terminal):
   ```bash
   source venv/bin/activate
   ```

3. **Prepare dataset**:
   - Place your raw dataset as `dataset.csv` in the `data/raw/` directory.

   If your file is named `dataset.csv`, run:
   ```bash
   make prepare-data
   ```
   If your file has a different name or location, run:
   ```bash
   make prepare-data DATASET=path/to/your/dataset.csv
   ```
   (The `DATASET` variable lets you specify a custom dataset path.)

## Usage

### Training Models

You can train models with different optimization algorithms:

1. **Train with default configuration**:
   ```bash
   make train
   ```

2. **Train with specific optimizer**:
   ```bash
   # For Adam optimizer
   make train-adam
   
   # For Momentum optimizer
   make train-momentum
   
   # For RMSprop optimizer
   make train-rmsprop
   ```

3. **Train with all optimizers**:
   ```bash
   make train-all
   ```

### Making Predictions

After training, you can use the models to make predictions:

1. **Predict with default model**:
   ```bash
   make predict
   ```

2. **Predict with specific model**:
   ```bash
   # For Adam model
   make predict-adam
   
   # For Momentum model
   make predict-momentum
   
   # For RMSprop model
   make predict-rmsprop
   ```

3. **Predict with all models**:
   ```bash
   make predict-all
   ```

### Complete Workflows

Run the entire process (prepare data, train, and predict) with:

```bash
# Using default configuration
make all-default

# Using Adam optimizer
make all-adam

# Using Momentum optimizer
make all-momentum

# Using RMSprop optimizer
make all-rmsprop

# Using all configurations
make all
```

### Data Visualization

To generate visualizations of your dataset:

If your file is named `dataset.csv` in the default location, run:
```bash
make visualize
```
If your file has a different name or location, run:
```bash
make visualize DATASET=path/to/your/dataset.csv
```
(The `DATASET` variable lets you specify a custom dataset path.)

### Cleanup

1. **Regular cleanup** (remove cached files and artifacts):
   ```bash
   make clean
   ```

2. **Deep cleanup** (remove virtual environment and all generated files):
   ```bash
   make deep-clean
   ```

### Help

For a full list of available commands:

```bash
make help
```

## Project Structure

```
├── Makefile           # Build automation
├── config/            # Configuration files for different optimizers
│   ├── network.json
│   ├── network_adam.json
│   ├── network_momentum.json
│   └── network_rmsprop.json
├── data/              # Data storage
│   ├── processed/     # Processed datasets
│   └── raw/           # Raw input dataset
├── models/            # Trained models
└── src/               # Source code
    ├── data_tools/    # Data processing and visualization tools
    │   ├── process_data_train.py
    │   ├── split_dataset.py
    │   └── visualizer.py
    ├── mlp_core/      # Core MLP implementation
    │   ├── dropout.py
    │   ├── layer.py
    │   ├── network.py
    │   ├── optimizers.py
    │   ├── parser.py
    │   └── perceptron.py
    ├── predict.py     # Prediction script
    └── train.py       # Training script
```