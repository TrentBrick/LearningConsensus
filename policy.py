""" Define controller """
import torch
import torch.nn as nn

class Policy(nn.Module):
    """ Decision making policy """
    def __init__(self, inputt, output, activation=nn.ReLU, output_activation=nn.Identity, 
    hiddens = [16,16]):
        super().__init__()

        sizes = [inputt]+hiddens+[output]

        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        layers = nn.Sequential(*layers)
        self.fc = layers

    def forward(self, *inputs):
        # returns argmax decision. 
        cat_in = torch.cat(inputs, dim=1)
        return torch.argmax(self.fc(cat_in))