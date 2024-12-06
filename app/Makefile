# Makefile for Transformer model

# Variables
PYTHON=python3
TRAIN_SCRIPT=src/train.py
MODEL_FILES=src/model.py src/positional_encoding.py src/dataset.py

# Default target: run training
all: train

# Target to train the model
train: $(MODEL_FILES)
	$(PYTHON) $(TRAIN_SCRIPT)

# Clean target to remove any compiled Python files
clean:
	find . -name "src/*.pyc" -delete
	find . -name "src/__pycache__" -delete

# Target to display help
help:
	@echo "Usage:"
	@echo "    make        Run training"
	@echo "    make clean  Remove compiled files"
	@echo "    make help   Show this help message"
