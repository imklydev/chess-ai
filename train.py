from device import device
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch

import time


def train_model(model, epochs, inputs, outputs):

    start_time = time.perf_counter()

    print("Training the model... (might take some time, get comfy!)")

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss().to(device)

    dataset = TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    if(device == torch.device("cpu")):
        print("\033[33mWARNING: Training on the CPU. Train times might be astronomical...\033[0m")

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
            print(f"Epoch {epoch+1}/{epochs}: loss = {sum(losses) / len(losses)}")


    end_time = time.perf_counter()

    print(f"Trained Chess AI from scratch, on the Lichess database, in {end_time - start_time:.1f}s")

    return model