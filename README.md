# Multilayer Perceptron (MLP) Project

This project aims to build a **Multilayer Perceptron (MLP)** neural network from scratch using Python, focusing on understanding key machine learning concepts such as **feedforward**, **backpropagation**, and **gradient descent**.

## Features

- **Data Splitting & Visualization**:
  - Split dataset into training and validation sets.
  - Visualize dataset distributions and relationships.

- **MLP Model**:
  - Build a **Multilayer Perceptron** with configurable layers and activation functions.
  - Train the model using **gradient descent** and **backpropagation**.

- **Prediction**:
  - Use the trained model to make predictions on new data.

## Installation

1. **Clone the repository**:
	```bash
	git clone link && cd multilayer-perceptron
	```

2. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```

3. **Setup the database**
	- Place your raw dataset into the data/raw/ directory.
	- Run split_dataset.py to process and split the data into data/processed/.

# Usage
1. **Data Splitting & Visualization**:
To shuffle, split the dataset, and visualize the data, run the following command:
```bash
python src/data_tools/split_dataset.py
```

This script will:
- Shuffle the dataset randomly.
- Split it into training and validation sets.
- Visualize the data distribution to help you understand the relationships in your dataset.

2. **Train the Model**:
To train the MLP model, run:

```bash
python src/scripts/train_mlp.py
```
- The model architecture is defined in config/architecture.txt.
- The trained model's weights will be saved to models/model_weights.json.

3. **Make Predictions**:
Once the model is trained, use the following command to make predictions:

```bash
python src/scripts/predict.py
```

This script will load the saved model weights and generate predictions based on the input data.

# Testing

You can run unit tests to ensure that your model and other modules are working correctly:

```bash
pytest tests/
```