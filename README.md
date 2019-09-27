# LearningConsensus

See google doc for notes: https://docs.google.com/document/d/1Za5KsQs4502lfNpebWS7CoPfJ8z-T8_H3LKt38wucfI/edit

RL algorithms are taken from FiredUp https://github.com/kashif/firedup 

TODO: 
* Implement the NN and RL algorithm (start with REINFORCE)
* Vectorize all the code so can have a batch of actions
* Get better visualizations of what the agents are doing 
* Better summary of the RL learning (reference the ppo algorithm for this)
* Implement PPO
* Parallelize code with MPI
* Implement different assumptions/scenarios in a modular way

config.py - has all of the configuration details (some commented out right now around the neural network not yet implemented)

main.py - runs the scripts and training loops

environment_and_agent_utils.py - does all of the heavy lifting for the environment and its actions, currently only the basic scenario implemented (see Google docs for the list of assumptions around the implementation)

nn.py - will hold all of the neural network details

ppo.py - Proximal Policy Optimization - implementation from FiredUp (PyTorch version of SpinningUp)

tester.py - way to experiment with different parts of the code


