import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader

def train(epochs, model, loss_fn, optimizer, train_dl, loss_min=None):

    for epoch in range(epochs):
        for xb, yb in train_dl:

            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()  # Derivetive
            optimizer.step()  # Update parameters (Our optimizer is stochastic gradient descent)
            optimizer.zero_grad()  # Reset gradients

        # Printing progress on each tenth number of epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch:[{epoch + 1}/{epochs}] Loss:{loss}")

        if loss_min is not None and loss < loss_min:
            print("Breaking because the minimum loss has been achieved...")
            print(float(loss))
            return

if __name__ == "__main__":

    inputs = np.array([
        [73, 67, 43],
        [91, 88, 64],
        [87, 134, 58],
        [102, 43, 37],
        [69, 96, 70],
        [74, 66, 43],
        [91, 87, 65],
        [88, 134, 59],
        [101, 44, 37],
        [68, 96, 71],
        [73, 66, 44],
        [92, 87, 64],
        [87, 135, 57],
        [103, 43, 36],
        [68, 97, 70],
    ], dtype=np.float32)
    targets = np.array([
        [56, 70],
        [81, 101],
        [119, 133],
        [22, 37],
        [103, 119],
        [57, 69],
        [80, 102],
        [118, 132],
        [21, 38],
        [104, 118],
        [57, 69],
        [82, 100],
        [118, 134],
        [20, 38],
        [102, 120]
        ], dtype=np.float32)

    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    loss_min = .6

    train_ds = TensorDataset(inputs, targets)
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)


    model = nn.Sequential(
        nn.Linear(3, 20),
        # nn.Sigmoid(),
        nn.ReLU(),
        nn.Linear(20, 2)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    train(100000, model, nn.functional.mse_loss, optimizer, train_dl, loss_min)

    preds = model(inputs)
    print(preds)
    print(targets)