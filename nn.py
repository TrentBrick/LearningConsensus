import torch
import numpy as np 

def randomActions(action_space):
    sel = np.random.choice(range(len(action_space)), 1)[0]
    action = action_space[sel]
    return action

def commitToValue(value):
    return 'commit_'+str(value)