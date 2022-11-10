import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Critic(nn.Module):

    def __init__(self, input_dims):
        """Init function

        Arguments:
            input_dims {int} -- Dimension of input vector (same as state dimension)
        """
        super(Critic, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dims, 64), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        """Forward function

        Arguments:
            x -- Input vector of the network
        """

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Recurrent_Actor(nn.Module):

    def __init__(self, input_dims, output_dims, hidden_dims):
        """Init function

        Arguments:
            input_dims {int} -- Dimension of input vector (same as state dimension)
            output_dims {int} -- Dimension of the output vector (same as action dimension)
        """
        super(Recurrent_Actor, self).__init__()
        self.layer1 = nn.RNN(input_dims, hidden_dims, batch_first=True)
        self.layer2 = nn.Sequential(nn.Linear(hidden_dims, 64), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(64, output_dims), nn.Tanh())

    def forward(self, x, hidden):
        """Forward function

        Arguments:
            x -- Input vector of the network
            hidden -- previous hidden recurrent layer output
        """

        out, hidden = self.layer1(x, hidden)
        x = self.layer2(out)
        x = self.layer3(x)
        return x, hidden

    def initHidden(self):
        return torch.zeros(1, 1, 64)