import torch
import torch.nn as nn
import numpy as np 

def randomActions(action_space):
    sel = np.random.choice(range(len(action_space)), 1)[0]
    action = action_space[sel]
    return action

# taken from FiredUp
class MLP(nn.Module):
    def __init__(self, layers, activation=torch.tanh, output_activation=None,
                 output_squeeze=False, use_bias=True):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer, bias=use_bias))
            if use_bias:
                nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class BasicPolicy(nn.Module):
    def __init__(self, action_dim, in_features, hidden_sizes,
     activation, output_activation, use_bias):
        super(BasicPolicy, self).__init__()

        self.logits = MLP(layers=[in_features]+list(hidden_sizes)+[action_dim],
                          activation=activation, output_activation=None, use_bias=use_bias)

    def forward(self, x):
        logits = self.logits(x)

        return logits 
        
'''def commitToValue(value):
    return 'commit_'+str(value)'''

