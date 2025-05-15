.PHONY: all setup clean train test visualize

PYTHON = python3
VENV = venv
DATA_DIR = data
MODELS_DIR = models
CONFIG_DIR = config

# Default target
all: prepare-data train predict

all-adam: prepare-data train-adam predict-adam
all-momentum: prepare-data train-momentum predict-momentum
all-rmsprop: prepare-data train-rmsprop predict-rmsprop

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	$(VENV)/bin/pip install -r requirements.txt
	@mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(MODELS_DIR)

# Clean build artifacts and temporary files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf training_history.png
	@echo "Clean complete."

# Deep clean (removes virtual environment and generated files)
deep-clean: clean
	@echo "Performing deep clean..."
	rm -rf $(VENV)
	rm -rf $(MODELS_DIR)/*
	@echo "Deep clean complete."

# Train the model with default configuration
train:
	@echo "Training model..."
	$(VENV)/bin/python src/train.py \
		--config $(CONFIG_DIR)/network.json \
		--data-train $(DATA_DIR)/processed/train_data.csv \
		--data-validation $(DATA_DIR)/processed/test_data.csv \
		--save $(MODELS_DIR)/model.pkl \
		--verbose

# Run with different configuration files
train-%:
	@echo "Training model with $* configuration..."
	$(VENV)/bin/python src/train.py \
		--config $(CONFIG_DIR)/network_$*.json \
		--data-train $(DATA_DIR)/processed/train_data.csv \
		--data-validation $(DATA_DIR)/processed/test_data.csv \
		--save $(MODELS_DIR)/model_$*.pkl \
		--verbose

predict:
	@echo "Predicting with model..."
	$(VENV)/bin/python src/predict.py \
		--model $(MODELS_DIR)/model.pkl \
		--data $(DATA_DIR)/processed/test_data.csv \
		--verbose


predict-%:
	@echo "Predicting with $* model..."
	$(VENV)/bin/python src/predict.py \
		--model $(MODELS_DIR)/model_$*.pkl \
		--data $(DATA_DIR)/processed/test_data.csv \
		--verbose

# Split and prepare dataset
prepare-data:
	@echo "Preparing dataset..."
	$(VENV)/bin/python src/data_tools/split_dataset.py --dataset_path $(DATA_DIR)/raw/dataset.csv --train_ratio 0.7

# Visualize data
visualize:
	@echo "Generating data visualizations..."
	$(VENV)/bin/python src/data_tools/visualize_data.py

# Run tests
test:
	@echo "Running tests..."
	$(VENV)/bin/pytest tests/

# Show help message
help:
	@echo "Multilayer Perceptron (MLP) Makefile"
	@echo "Available targets:"
	@echo "  all          : Setup environment and train model (default)"
	@echo "  setup        : Create virtual environment and install dependencies"
	@echo "  clean        : Remove cached files and artifacts"
	@echo "  deep-clean   : Remove virtual environment and all generated files"
	@echo "  train        : Train model with default configuration"
	@echo "  train-<name> : Train model with <name>.json configuration"
	@echo "  prepare-data : Process and split raw dataset"
	@echo "  visualize    : Generate data visualizations"
	@echo "  test         : Run tests"
	@echo "  help         : Show this help message"