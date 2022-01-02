# LearningConsensus

Supervised by Dr. Kartik Nayak, final class project turned research project. Investigated the ability for deep reinforcement learning agents to discover and prove BFT consensus protocols. This was a great way to learn more about reinforcement learning but the tasks were too difficult for the agents to learn.

Thanks to Yash Patil and Rohan Reddy for their collaboration with this work. 

RL algorithms are taken from FiredUp https://github.com/kashif/firedup 

See `TrentonBricken_Learning_Consensus.pdf` for a writeup of an early stage of this project. 

# Code Summary

Note that the code is quite undocumented at this point. 

config.py - has all of the configuration details (some commented out right now around the neural network not yet implemented)

trainer.py - runs the scripts and training loops

environment_and_agent_utils.py - does all of the heavy lifting for the environment and its actions, currently only the basic scenario implemented (see Google docs for the list of assumptions around the implementation)

nn.py - will hold all of the neural network details

es.py - evolutionary search, uses David Ha's library. 

ppo.py - Proximal Policy Optimization - implementation from FiredUp (PyTorch version of SpinningUp)

tester.py - way to experiment with different parts of the code