import numpy as np
import torch
import torch.nn as nn

from src.config import ACTOR_SIZE_1, ACTOR_SIZE_2, CRITIC_SIZE_2, \
    CRITIC_SIZE_1


def init_hidden_layers(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class ActorNet(nn.Module):

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        # self.bn_1 = nn.BatchNorm1d(state_size)
        self.lin_1 = nn.Linear(state_size, ACTOR_SIZE_1)
        self.bn_2 = nn.BatchNorm1d(ACTOR_SIZE_1)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(ACTOR_SIZE_1, ACTOR_SIZE_2)
        self.bn_3 = nn.BatchNorm1d(ACTOR_SIZE_2)
        self.relu_2 = nn.ReLU()
        self.lin_3 = nn.Linear(ACTOR_SIZE_2, action_size)
        self.tanh = nn.Tanh()

        self.initialize_params()

    def forward(self, x):
        # x = self.bn_1(x)
        x = self.lin_1(x)
        x = self.bn_2(x)
        x = self.relu_1(x)
        x = self.lin_2(x)
        x = self.bn_3(x)
        x = self.relu_2(x)
        x = self.lin_3(x)
        x = self.tanh(x)
        return x

    def initialize_params(self):
        self.lin_1.weight.data.uniform_(*init_hidden_layers(self.lin_1))
        self.lin_2.weight.data.uniform_(*init_hidden_layers(self.lin_2))
        self.lin_3.weight.data.uniform_(-3e-3, 3e-3)


class CriticNet(nn.Module):

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        # self.bn_1 = nn.BatchNorm1d(state_size)
        self.lin_1 = nn.Linear(state_size, CRITIC_SIZE_1)
        self.bn_2 = nn.BatchNorm1d(CRITIC_SIZE_1)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(CRITIC_SIZE_1 + action_size, CRITIC_SIZE_2)
        self.relu_2 = nn.ReLU()
        self.bn_3 = nn.BatchNorm1d(CRITIC_SIZE_2)
        self.lin_3 = nn.Linear(CRITIC_SIZE_2, 1)

        self.initialize_params()

    def forward(self, state, action):

        # concat = self.bn_1(state)
        concat = self.lin_1(state)
        concat = self.bn_2(concat)
        concat = self.relu_1(concat)
        concat = torch.cat([concat, action], dim=1)
        concat = self.lin_2(concat)
        concat = self.bn_3(concat)
        concat = self.relu_2(concat)
        concat = self.lin_3(concat)

        return concat

    def initialize_params(self):
        self.lin_1.weight.data.uniform_(*init_hidden_layers(self.lin_1))
        self.lin_2.weight.data.uniform_(*init_hidden_layers(self.lin_2))
        self.lin_3.weight.data.uniform_(-3e-3, 3e-3)
