from unityagents import UnityEnvironment

from src.config import UNITY_ENV_PATH


class UnityRun:
    def __init__(self):

        self.env = UnityEnvironment(UNITY_ENV_PATH)

    def __enter__(self):
        return self.env

    def __exit__(self):
        self.env.close()
