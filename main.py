import click
import logging
import neptune.new as neptune
import numpy as np
import os
import torch
from unityagents import UnityEnvironment

from src.agent import Agent
from src.config import MAX_EPOCH, CHECKPOINT_SAVE_PATH, LEARNING_RATE, GAMMA, TAU, BATCH_SIZE
from src.structs import env_feedback


@click.group(chain=True, invoke_without_command=True)
def run():
    logging.info("Running the ml-monitoring project")


@run.command("train", short_help="Train the reinforcement learning model")
@click.option(
    "-l",
    "--log",
    help="Flag whether the experiment should be logged to neptune.ai",
    required=False
)
def train(log: bool):
    env, agent, scores, brain_name = setup_environment()

    if log:
        neptune_log = log_to_neptune()
    #
    # epsilon_space = np.concatenate([
    #     np.linspace(EPS_MAX, EPS_MIN, EXPLORATORY_EPOCHS),
    #     np.repeat(EPS_MIN, (MAX_EPOCH - EXPLORATORY_EPOCHS))
    # ])

    for episode in range(MAX_EPOCH):
        env_info = env.reset(train_mode=True)[brain_name]
        start_state = env_info.vector_observations[0]
        # eps = epsilon_space[episode]
        score = act_during_episode(agent, env, start_state, brain_name, 0.01)

        if log:
            neptune_log['score'].log(score)
        scores.append(score)
        print(
            f"Episode {episode} | Score = {score} | Max score = {np.max(scores)} | Avg = {np.mean(scores[-100:]):.2f}")

    if log:
        neptune_log.stop()
    torch.save(agent.actor_network_local.state_dict(), CHECKPOINT_SAVE_PATH)
    env.close()


@run.command("evaluate", short_help="Run the agent based on saved weights")
def evaluate():
    logging.info("Setting up the environment for evaluation")
    env, agent, _, brain_name = setup_environment(read_saved_model=True)

    env_info = env.reset(train_mode=False)[brain_name]
    start_state = env_info.vector_observations[0]

    score = act_during_episode(agent, env, start_state, brain_name, EPS_MIN)
    print(f"Evaluation score = {score}")
    env.close()


def setup_environment(read_saved_model=False):
    env = UnityEnvironment(
        file_name="/Users/wojciech.szenic/PycharmProjects/rl-continuous-control/unity_env_single/Reacher.app")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    state = env_info.vector_observations[0]

    agent = Agent(state_size=len(state), action_size=brain.vector_action_space_size, read_saved_model=read_saved_model)

    scores = []

    return env, agent, scores, brain_name


def act_during_episode(agent, env, state, brain_name, eps):
    score = 0
    while True:
        action = agent.act(state, eps)
        env_info = env.step(action.numpy())[brain_name]
        env_response = env_feedback(state, action, env_info.rewards[0], env_info.vector_observations[0],
                                    env_info.local_done[0])
        agent.step(env_response)

        score += env_response.reward
        state = env_response.next_state
        if env_response.done:
            break
    return score


def log_to_neptune():
    neptune_run = neptune.init(
        project="wsz/RL-AgentCritic",
        api_token=os.getenv('NEPTUNE_TOKEN')
    )

    neptune_run['parameters'] = {
        'LEARNING_RATE': LEARNING_RATE,
        'GAMMA': GAMMA,
        'TAU': TAU,
        'BATCH_SIZE': BATCH_SIZE
    }
    return neptune_run


if __name__ == "__main__":
    run()
