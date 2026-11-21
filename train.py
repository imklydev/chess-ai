from device import device
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_model(model, epochs, inputs, outputs):

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss().to(device)

    dataset = TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        losses = []

        for X, y in dataloader:
            prediction = model(X)
            loss = loss_fn(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0:
                losses.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss = {sum(losses) / len(losses)}")

    return model