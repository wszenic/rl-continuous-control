from collections import namedtuple

env_feedback = namedtuple('env_feedback', ('state', 'action', 'reward', 'next_state', 'done'))
