import torch
import torch.nn as nn


class ActorNet(nn.Module):

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        self.lin_1 = nn.Linear(state_size, 256)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(256, 256)
        self.relu_2 = nn.ReLU()
        self.lin_3 = nn.Linear(256, action_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.relu_1(x)
        x = self.lin_2(x)
        x = self.relu_2(x)
        x = self.lin_3(x)
        x = self.tanh(x)
        return x


class CriticNet(nn.Module):

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        # state block
        self.state_lin_1 = nn.Linear(state_size, 16)
        self.state_relu_1 = nn.ReLU()
        self.state_lin_2 = nn.Linear(16, 32)
        self.state_relu_2 = nn.ReLU()

        # action block
        self.action_lin_1 = nn.Linear(action_size, 32)

        # post-concatenation blocks
        self.concat_lin_1 = nn.Linear(64, 256)
        self.concat_relu_1 = nn.ReLU()
        self.concat_lin_2 = nn.Linear(256, 256)
        self.concat_relu_2 = nn.ReLU()
        self.concat_lin_3 = nn.Linear(256, 1)

    def forward(self, state, action):
        state = self.state_lin_1(state)
        state = self.state_relu_1(state)
        state = self.state_lin_2(state)
        state = self.state_relu_2(state)

        action = self.action_lin_1(action.float())

        concat = torch.cat([state, action], dim=1)

        concat = self.concat_lin_1(concat)
        concat = self.concat_relu_1(concat)
        concat = self.concat_lin_2(concat)
        concat = self.concat_relu_2(concat)
        concat = self.concat_lin_3(concat)

        return concat
