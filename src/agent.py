from collections import namedtuple

import numpy as np
import random
import torch

from src.config import UPDATE_INTERVAL, BATCH_SIZE, BUFFER_SIZE, GAMMA, TAU, LEARNING_RATE, CHECKPOINT_SAVE_PATH
from src.memory import ReplayBuffer
from src.network import ActorNet
from src.network import CriticNet


class Agent:

    def __init__(self, state_size: int, action_size: int, read_saved_model=False):

        self.state_size = state_size
        self.action_size = action_size

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.actor_network_local = ActorNet(state_size, action_size).to(device)
        self.actor_network_target = ActorNet(state_size, action_size).to(device)

        self.critic_network_local = CriticNet(state_size, action_size).to(device)
        self.critic_network_target = CriticNet(state_size, action_size).to(device)

        if read_saved_model:
            saved_model = torch.load(CHECKPOINT_SAVE_PATH)
            self.actor_network_local.load_state_dict(saved_model)
            self.critic_network_local.load_state_dict(saved_model)

        self.actor_optimizer = torch.optim.Adam(self.actor_network_local.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic_network_local.parameters(), lr=LEARNING_RATE)

        self.gamma = GAMMA
        self.tau = TAU

        self._step = 0
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device)
        self.env_feedback = namedtuple('env_feedback', ('state', 'action', 'reward', 'next_state', 'done'))

    def step(self, env_data):
        self.memory.add(
            state=env_data.state,
            action=env_data.action,
            reward=env_data.reward,
            next_state=env_data.next_state,
            done=env_data.done,
        )

        self._step = (self._step + 1) % UPDATE_INTERVAL
        if self._step == 0:
            if len(self.memory) > BATCH_SIZE:
                experience_replay = self.memory.sample()

                self.learn(self.env_feedback(*experience_replay))

    def act(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_values = self.__get_state_action_values(state)
        return action_values

    def learn(self, env_data):
        target_actions = self.actor_network_target(env_data.next_state)
        y = env_data.reward + GAMMA * self.critic_network_target(
            env_data.next_state, target_actions
        )

        # critic
        critic_value = self.critic_network_local(env_data.state, env_data.action)
        critic_loss = torch.mean(torch.square(y - critic_value))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.__soft_update(self.critic_network_local, self.critic_network_target)

        # actor
        actor_actions = self.actor_network_local(env_data.state)
        critic_value = self.critic_network_local(env_data.state, actor_actions)
        actor_loss = -torch.mean(critic_value)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.__soft_update(self.actor_network_local, self.actor_network_target)

    def __get_state_action_values(self, state):
        self.actor_network_local.eval()
        with torch.no_grad():
            actions_values = self.actor_network_local(state)
        self.actor_network_local.train()

        return actions_values

    def __get_epsilon_greedy_action(self, action_values, eps):
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def __soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)