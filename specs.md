
# Multilayer Perceptron – Project Specifications

## Context

The goal of this project is to implement a simple Multilayer Perceptron (MLP) neural network from scratch, without relying on machine learning frameworks such as TensorFlow or PyTorch. The implementation will include key components such as data visualization, dataset splitting, forward propagation, backpropagation, and gradient descent.

This project serves both as a practical introduction to core neural network concepts and a demonstration of understanding how neural networks learn from data.

## Objectives

- Understand and implement the fundamental algorithms of a Multilayer Perceptron:
  - Feedforward
  - Backpropagation
  - Gradient Descent
- Develop three independent programs:
  1. A program to split the dataset into training and validation sets
  2. A program to train the MLP using a configuration file for its architecture
  3. A program to make predictions using the trained model
- Visualize the data before training
- Evaluate the model using a validation set

## Task Breakdown

### 0. Study Core Concepts

**0.1. Understand Feedforward**
- Learn how data flows through the network from input to output.
- Study activation functions and their roles (e.g., ReLU, sigmoid, softmax).

**0.2. Understand Backpropagation**
- Learn how the error is calculated and propagated backward through the network.
- Study the chain rule and how it applies to computing gradients.

**0.3. Understand Gradient Descent**
- Learn how weights and biases are updated using the computed gradients.
- Understand the impact of learning rate and the role of epochs.

### 1. Dataset Splitting & Visualization (`split_dataset.py`)

**1.1. Load Data**
- Read the raw dataset from a CSV file using `pandas`.

**1.2. Visualize Data**
- Plot input features using scatter plots or histograms.
- Optionally visualize class distribution if labels are present.

**1.3. Shuffle and Split Data**
- Shuffle the dataset randomly.
- Split into training set (80%) and validation set (20%).

**1.4. Save Split Datasets**
- Write `train.csv` and `validation.csv` to disk using `pandas`.

### 2. Training Program (`train_mlp.py`)

**2.1. Parse Network Architecture**
- Read and parse a text file describing the architecture (number of layers, units per layer, activation functions).

**2.2. Initialize Network**
- Create weight matrices and bias vectors with random initialization for each layer.

**2.3. Load and Preprocess Training Data**
- Load `train.csv` and `val.csv`.
- Normalize input features.
- One-hot encode output labels if it's a classification task.

**2.4. Implement Feedforward Pass**
- Apply linear transformation and activation function at each layer.
- Output final predictions from the network.

**2.5. Implement Backpropagation**
- Compute error between predictions and true labels.
- Propagate error backward through the network.
- Calculate gradients for each layer.

**2.6. Implement Gradient Descent**
- Update weights and biases using gradients and a learning rate.

**2.7. Train the Network**
- Loop over several epochs.
- Compute and log training loss and accuracy.
- Validate on the validation set at intervals.

**2.8. Save Model Weights**
- Store trained weights and biases in a JSON file (`model_weights.json`).

### 3. Prediction Program (`predict.py`)

**3.1. Load Trained Weights**
- Read the JSON file containing saved model weights and biases.

**3.2. Load Input Data**
- Accept new input(s) from file or command line.

**3.3. Preprocess Input**
- Normalize input features based on training stats (if needed).

**3.4. Run Feedforward Pass**
- Use the same architecture to compute predictions on the new input.

**3.5. Output Predictions**
- Print predictions to the console or save them to a file.


### 4. Bonuses

**4.1. Add optimizers**
- Add few optimizers such as Adam, RMSProp, Momentum

**4.2. Display multiple learning curves**
- The program can parse and train multiple models at once and display a graph that compare them

**4.3. History of metrics**
- Do a history of metrics that can be saved in .json file

**4.4. Implementation of early stop**
- Implement an early stop that is configurable inside network in the config

**4.5. Evaluate the learning phase with multiple metrics**
- Add few more metrics such as F1 score, recall, precision, and accuracies


## Development Workflow

This workflow is designed not only to ensure clean, incremental development, but also to help you build strong habits that mirror professional practices.

Even though this is a **school project**, the workflow follows the same structure used at **company**. Practicing this structure helps reinforce industry-standard habits and makes the transition to production-level collaboration smoother.

### Branching Strategy

- Each **small task** in the Task Breakdown should have its **own dedicated Git branch**. (Except for the 0 part)
  - Branch name format: `feature/task-name`, for example: `feature/parse-architecture`
- Only work on **one task per branch**.

### Development Steps

1. **Create a branch** for the task you’re working on:
   ```bash
   git checkout -b feature/task-name
   ```

2. **Implement the task** and commit your changes regularly:
   ```bash
   git add .
   git commit -m "Implement task-name"
   ```

3. **Push your branch** to the remote repository:
   ```bash
   git push origin feature/task-name
   ```

4. **Open a Pull Request (PR)**:
   - Title: `Task: Short description of the task`
   - Description: Link to the task from the spec and summarize what was done.

5. **Review**:
   - Use **GitHub Copilot** or any other code review assistant for initial feedback. (It's a solo project so no humans can review the code)

6. **Merge**:
   - Once approved, merge the PR into the `main` branch.
   - Delete the feature branch to keep the repo clean.

### Notes
- Always pull the latest changes from `main` before starting a new task:
  ```bash
  git checkout main
  git pull origin main
  ```
- Keep commits small and meaningful.

This approach ensures incremental progress, code review quality, and easier collaboration or debugging.

## Code Structure

Here is the directory structure of the project:

```bash
mlp-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
│
├── models/
│   └── model_weights.json
│
├── src/
│   ├── data_tools/
│   │   ├── process_data_train.py
│   │   ├── split_dataset.py
│   │   └── visualizer.py
│   │
│   ├── mlp_core/
│   │   ├── __init__.py
│   │   ├── perceptron.py
│   │   ├── network.py
│   │   ├── layer.py
│   │   ├── optimizers.py (if done with bonus)
│   │   └── parser.py
│   ├── train.py
│   └── predict.py
│
├── config/
│   └── architecture.json
│
├── specs.md
├── README.md
├── requirements.txt
├── Makefile
└── .gitignore
```

This structure splits the code into logical groups:
- **`data_tools/`**: Functions related to dataset manipulation and visualization.
- **`mlp_core/`**: The core of the MLP implementation, including the network, layers, and utility functions.
- **`scripts/`**: The scripts to run the model for training or prediction.

This will help you maintain a clean and organized project while developing each part incrementally.
