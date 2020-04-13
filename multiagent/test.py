import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multi_utils import MultiAgentActionSpace, MultiAgentObservationSpace

n=3
action_space = MultiAgentActionSpace([spaces.Discrete(3) for _ in range(n)])
        # configure observation space
observation_space = MultiAgentObservationSpace([spaces.Box(0, 2, (n,), dtype=np.uint8) for _ in range(n)])

obs_dim = observation_space[0].shape
act_dim = action_space[0].shape
print(action_space)
print(observation_space[0].low)
print("obs_dim: " + str(obs_dim))
print("act_dim: " + str(act_dim))