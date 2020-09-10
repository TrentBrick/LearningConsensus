""" Define controller """
import torch
import torch.nn as nn
import numpy as np

class Policy(nn.Module):
    """ Decision making policy """
    def __init__(self, inputt, output, activation=nn.ReLU, output_activation=nn.Identity, 
    hiddens = [16,8]):
        super().__init__()

        print('inputs to nn', inputt, output, hiddens)

        inputt = int(inputt.shape[0]) * 3

        sizes = [inputt]+hiddens+[output.n]

        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        layers = nn.Sequential(*layers)
        self.fc = layers

    def forward(self, *inputs):
        # returns argmax decision. 
        #print('inputs are', inputs)

        hots = []
        for i in inputs[0]: 
            temp = np.zeros(3)
            temp[int(i)] = 1
            hots.append(temp)

        hots = torch.Tensor(hots).view(1,-1)
        #print('hots is', hots, hots.shape)
        #cat_in = torch.cat(hots, dim=0).unsqueeze(0)
        #print('into nn', cat_in)
        out = self.fc(hots)
        #print('out of nn', out)
        return torch.argmax(out)