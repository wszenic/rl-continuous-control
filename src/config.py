# Paths
UNITY_ENV_LOCATION = "../unity_env_single/Reacher.app'"
CHECKPOINT_SAVE_PATH = "./checkpoints/model_checkpoint.pth"

MAX_EPOCH = 2000

# Hyperparams

ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
GAMMA = 0.99
TAU = 0.001
# TAU = 0.02

# Network sizes
ACTOR_MID_1 = 64
ACTOR_MID_2 = 64

CRITIC_STATE_1 = 32
CRITIC_STATE_2 = 32
CRITIC_ACTION_1 = 32

CRITIC_CONCAT_1 = 64
CRITIC_CONCAT_2 = 64

# Learning
BATCH_SIZE = 64
UPDATE_INTERVAL = 4

# Memory
BUFFER_SIZE = int(1e6)

# Noise
NOISE_STD = 0.2
