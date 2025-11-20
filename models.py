import torch


class Model(torch.nn.Module):
    """
    A simple 3 layer model inheriting from torch's standard nn
    """
    def __init__(self):
        # call init of super class we are inheriting from
        super().__init__()
        # totally arbitrarily choosing the network architecture here
        # we have 25 inputs, and 1 output
        # and let's start with basic linear layers
        self.fc1 = torch.nn.Linear(25, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.relu(self.fc1(inp))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
