# Define XOR input and target data
from BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from Linear import Linear
from Sequential import Sequential
from Sigmoid import Sigmoid

import numpy as np


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Define the model architecture
model = Sequential()
model.add(Linear(2, 2))
model.add(Sigmoid())
model.add(Linear(2, 1))
model.add(Sigmoid())

# Train the model
learning_rate = 0.1
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    output = model.forward(X)
    
    print(X.shape)
    
    # Compute loss
    loss_layer = BinaryCrossEntropyLoss()
    print(output.shape,Y.shape)
    loss = np.mean(loss_layer.forward(output, Y))
    
    # Backward pass
    grad_loss = loss_layer.backward("")
    print("Shape of grad_loss:", grad_loss.shape)
    print("Shape of forward pass output:", output.shape)
    model.backward(grad_loss)
    
    # Gradient descent update
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            layer.weights -= learning_rate * getattr(layer, 'grad_weights', 0)
            layer.bias -= learning_rate * getattr(layer, 'grad_bias', 0)
    
    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss {loss}")

# Save the trained weights
model.save_weights("XOR_solved.w")
