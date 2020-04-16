import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multi_utils import MultiAgentActionSpace, MultiAgentObservationSpace

n=3
# action_space = MultiAgentActionSpace([spaces.Discrete(3) for _ in range(n)])
        # configure observation space
action_space = spaces.Discrete(3)
observation_space = spaces.Box(0, 2, (n,), dtype=np.uint8)

obs_dim = observation_space.shape
act_dim = action_space.shape
print(action_space)
print(observation_space)
print("obs_dim: " + str(obs_dim))
print("act_dim: " + str(act_dim))