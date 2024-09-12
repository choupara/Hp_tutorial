#!/usr/bin/env python
# coding: utf-8

# # Example for using OmniOpt
#
# source code taken from: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# parameters under consideration:#
# 1. batch size
# 2. epochs
# 3. size output layer 1
# 4. size output layer 2

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import argparse

# parsing hpyerparameters as arguments
parser = argparse.ArgumentParser(description="Demo application for OmniOpt for hyperparameter optimization, example: neural network on MNIST fashion data.")

parser.add_argument("--out-layer1", type=int, help="the number of outputs of layer 1", default = 512)
parser.add_argument("--out-layer2", type=int, help="the number of outputs of layer 2", default = 512)
parser.add_argument("--batchsize", type=int, help="batchsize for training", default = 64)
parser.add_argument("--epochs", type=int, help="number of epochs", default = 5)

args = parser.parse_args()

batch_size = args.batchsize
epochs = args.epochs
num_nodes_out1 = args.out_layer1
num_nodes_out2 = args.out_layer2

# Download training data from open data sets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open data sets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, out1, out2):
        self.o1 = out1
        self.o2 = out2
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, out1),
            nn.ReLU(),
            nn.Linear(out1, out2),
            nn.ReLU(),
            nn.Linear(out2, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork(out1=num_nodes_out1, out2=num_nodes_out2).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    #print statement esp. for OmniOpt (single line!!)
    print(f"RESULT: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")