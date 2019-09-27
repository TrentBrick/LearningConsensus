
from nn import model

def decideAction(party_state):
    # given the state, put into the NN what the action that the agent decides to take is
    # this state is agent specific
    # returns the action

    action = model(party_state)

    return action