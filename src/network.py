import torch
import torch.nn as nn

from src.config import ACTOR_MID_1, ACTOR_MID_2, CRITIC_STATE_1, CRITIC_STATE_2, CRITIC_ACTION_1, CRITIC_CONCAT_2, \
    CRITIC_CONCAT_1


class ActorNet(nn.Module):

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        self.bn_1 = nn.BatchNorm1d(state_size)
        self.lin_1 = nn.Linear(state_size, ACTOR_MID_1)
        self.bn_2 = nn.BatchNorm1d(ACTOR_MID_1)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(ACTOR_MID_1, ACTOR_MID_2)
        self.bn_3 = nn.BatchNorm1d(ACTOR_MID_2)
        self.relu_2 = nn.ReLU()
        self.lin_3 = nn.Linear(ACTOR_MID_2, action_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.bn_1(x)
        x = self.lin_1(x)
        x = self.bn_2(x)
        x = self.relu_1(x)
        x = self.lin_2(x)
        x = self.bn_3(x)
        x = self.relu_2(x)
        x = self.lin_3(x)
        x = self.tanh(x)
        return x


class CriticNet(nn.Module):

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        # state block
        self.bn_1 = nn.BatchNorm1d(state_size)
        self.state_lin_1 = nn.Linear(state_size, CRITIC_STATE_1)
        self.bn_2 = nn.BatchNorm1d(CRITIC_STATE_1)
        self.state_relu_1 = nn.ReLU()
        self.state_lin_2 = nn.Linear(CRITIC_STATE_1, CRITIC_STATE_2)
        self.bn_3 = nn.BatchNorm1d(CRITIC_STATE_2)
        self.state_relu_2 = nn.ReLU()

        # action block
        self.action_lin_1 = nn.Linear(action_size, CRITIC_ACTION_1)

        # post-concatenation blocks
        self.concat_lin_1 = nn.Linear(CRITIC_ACTION_1 + CRITIC_STATE_2, CRITIC_CONCAT_1)
        self.concat_relu_1 = nn.ReLU()
        self.concat_lin_2 = nn.Linear(CRITIC_CONCAT_1, CRITIC_CONCAT_2)
        self.concat_relu_2 = nn.ReLU()
        self.concat_lin_3 = nn.Linear(CRITIC_CONCAT_2, 1)

    def forward(self, state, action):
        state = self.bn_1(state)
        state = self.state_lin_1(state)
        state = self.bn_2(state)
        state = self.state_relu_1(state)
        state = self.state_lin_2(state)
        state = self.bn_3(state)
        state = self.state_relu_2(state)

        action = self.action_lin_1(action.float())

        concat = torch.cat([state, action], dim=1)

        concat = self.concat_lin_1(concat)
        concat = self.concat_relu_1(concat)
        concat = self.concat_lin_2(concat)
        concat = self.concat_relu_2(concat)
        concat = self.concat_lin_3(concat)

        return concat
