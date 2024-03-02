import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Due to the conflict of numpy and torch with the file of "libiomp5md.dll"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('line_fit_data.csv')

# Extract x and y values from the dataframe
X = data['x'].values.reshape(-1, 1)
y = data['y'].values

# Convert the data to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegressionModel(input_dim=1, output_dim=1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Lists to store loss values
losses = []

# Train the model
num_epochs = 2000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor.unsqueeze(1))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store the loss value
    losses.append(loss.item())
    
    # Print the loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, get the weights and bias
w, b = model.linear.weight.data.item(), model.linear.bias.data.item()

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
plt.savefig('loss_curve.png')


