import torch
import torch.nn as nn
import torch.optim as optim
from src.model import TransformerModel
from src.positional_encoding import PositionalEncoding
from src.dataset import get_data


# Hyperparameters
input_size = 1000  # vocab size
output_size = 1000  # vocab size
epochs = 20

# Initialize model, optimizer, loss function
model = TransformerModel(input_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Load data
train_data, train_labels = get_data()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_data, train_labels)
    loss = criterion(outputs.view(-1, output_size), train_labels.view(-1))
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
