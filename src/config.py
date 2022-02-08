# Paths
UNITY_ENV_LOCATION = "../unity_env_single/Reacher.app'"
CHECKPOINT_SAVE_PATH = "./checkpoints/model_checkpoint.pth"

MAX_EPOCH = 200

# Hyperparams

LEARNING_RATE = 1e-4
GAMMA = 0.99
TAU = 1e-3
EPS_MAX = 1
EPS_MIN = 0.01

# Learning
BATCH_SIZE = 64
UPDATE_INTERVAL = 4

# Optimizer
MOMENTUM = 0.95

# Memory
BUFFER_SIZE = int(1e6)
