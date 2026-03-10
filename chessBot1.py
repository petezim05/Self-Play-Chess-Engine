import torch
import torch.nn as nn
import torch.nn.functional as F

class RecConNet(nn.Module):

    def __init__(self):
        super().__init__()
        #convolutional layers
        self.conv1 = nn.Conv2d(18, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5,padding=2)

        #linear layers
        self.lin1 = nn.Linear(128, 64, bias=True)
        self.lin2 = nn.Linear(64, 64, bias=True)

        #output
        self.output = nn.Linear(64, 1, bias=True)

    def forward(self, x):
        x = x
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size= 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size= 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return F.tanh(self.output(x))  