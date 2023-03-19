# Link To The Video:
# https://www.youtube.com/watch?v=m_tkL7DufPk&list=PLyMom0n-MBroupZiLfVSZqK5asX8KfoHL&index=2

import numpy as np
import torch

# Inputs : Humidty, temperature and rainfall values respectively, independent from their units.
# Targets: Yield of apples and oranges in tonnes respectively.
INPUTS = [
    [73, 67, 43],
    [94, 88, 64],
    [87, 134, 58],
    [102, 43, 27],
    [69, 96, 70]
]
TARGETS = [
    [56, 70],
    [81, 101],
    [119, 133],
    [22, 37],
    [103, 119],
]

def model(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    return x @ w.t() + b

def mse_loss(preds, targets):
    diff = targets - preds
    return torch.sum(diff * diff) / diff.numel()

def train(targets, weights, biases, epochs):

    print("Training...")
    print(f"Epochs: {epochs}")

    for _ in range(epochs):

        # Predict
        preds = model(inputs, weights, biases)

        # Calculate the loss
        loss = mse_loss(preds, targets)

        # Derive loss
        loss.backward()
        
        # Adjust weights and biases
        with torch.no_grad():
            weights -= weights.grad * 1e-5      
            biases -= biases.grad * 1e-5

        # Reset weight and bias gradients
        weights.grad.zero_()
        biases.grad.zero_()


if __name__ == "__main__":

    epochs = 1000000

    inputs = torch.from_numpy(np.array(INPUTS, dtype=np.float32))
    targets = torch.from_numpy(np.array(TARGETS, dtype=np.float32))

    weights = torch.randn(2, 3, requires_grad=True)
    biases = torch.randn(2, requires_grad=True)

    # Initial Predictions
    preds = model(inputs, weights, biases)  # Predict
    print("Initial predictions:")
    print(preds)

    train(targets, weights, biases, epochs)

    # Predictions After Training - Comparison To The Target
    preds = model(inputs, weights, biases)  # Predict
    print("Predictions after training:")
    print(preds)
    print("Real values:")
    print(targets)

