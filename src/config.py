# Paths
UNITY_ENV_LOCATION = "unity_env_single/Reacher.app"
CHECKPOINT_SAVE_PATH = "./checkpoints/model_checkpoint.pth"

# Epochs
MAX_EPOCH = 300
EPOCHS_WITH_NOISE = 50

# Hyperparams

ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
GAMMA = 0.9
TAU = 1e-2
# TAU = 0.02

# Network sizes
ACTOR_SIZE_1 = 128
ACTOR_SIZE_2 = 128

CRITIC_SIZE_1 = 128
CRITIC_SIZE_2 = 128

# Learning
BATCH_SIZE = 256

# Memory
BUFFER_SIZE = int(1e6)

# Noise
NOISE_STD = 0.01

# Checkpoint
CHECKPOINT_EVERY = 50
