import numpy as np 
import torch
from config import *
from nn import *

def toOneHot(state):
    oh = []
    #print('full state for an agent', state)
    for s in state:
        #print('state for one hot',s)
        oh.append(oneHotMapper[s])
    oh = np.asarray(oh).T.flatten() # now each column is one of the states.
    #convert ot pytorch tensor: 
    oh = torch.from_numpy(oh).float().to(device)
    return oh

class Agent:
    def __init__(self, isByzantine, agentID, byzantine_inds=None):
        self.isByzantine = isByzantine
        self.agentID = agentID
        if isByzantine:
            #self.brain = randomActions
            self.brain = byz_policy
        else: 
            #self.brain = randomActions
            self.brain = honest_policy
        self.actionSpace = getActionSpace(isByzantine, byzantine_inds) 
        init_val = np.random.choice(commit_vals, 1)[0]
        self.initVal = init_val
        initState = [init_val]
        for a in range(num_agents-1):
            initState.append(null_message_val)
        self.initState = initState
        self.state = self.initState
        self.committed_value = False

    def commitValue(self, value):
        self.committed_value = value

    def chooseAction(self, temperature, forceCommit=False):
        if self.committed_value == True: # dont allow to take any actions. just keep committing.  
            self.action = 'commit_'+str(self.committed_value)
            action_logprob = 0
        else: 
            # look at the current state and decide what action to take. 
            oh = toOneHot(self.state) # each column is one of the states. 
            #making a decision:
            #self.action = self.brain(self.actionSpace)
            # expects softmax to be applied to network first.
            logits = self.brain(oh)

            if forceCommit:
                if not self.isByzantine: #if it is an honest agent
                    commit_inds = [ ind for ind, a in enumerate(self.actionSpace) if 'commit' in a ]
                    logits = logits[commit_inds] # make it so that the only logits are those for committing. 

            real_logprobs = torch.log(torch.nn.functional.softmax(logits, dim=0)) # currently not vectorized
            #should be able to apply sampling without computing this twice... 
            temperature_probs =  torch.nn.functional.softmax(logits/temperature, dim=0) 
            action_ind = torch.multinomial(temperature_probs, 1, replacement=False) # returns 1 sample per row. 
            action_logprob = real_logprobs[action_ind]
            self.action = self.actionSpace[action_ind]

            if self.action.split('_')[0]== 'commit': # checking for a commit. 
                self.commitValue(self.action.split('_')[1])

        return self.state, self.action, action_logprob
            
def updateStates(agent_list):
    #look at all agent actions and update the state of each to accomodate actions

    for reciever in agent_list:
        new_state = [reciever.initVal]
        for actor in agent_list:
            if actor == reciever: # check if agent committed. 
                continue
            # dont need to check if committed a value as already will prevent 
            # from taking any actions other than no send. 
            new_state.append( actionEffect(actor.action, reciever.agentID) )

        reciever.state = new_state

def actionEffect(action, receiver_id):
    # return the effects of a particular action

    if action == 'no_send' or action.split('_')[0]=='commit':
        return null_message_val

    elif 'to_all' in action:
        return int(action.split('_')[-1])
    
    elif 'agent-'+str(receiver_id) in action:
        # getting the first value of this. 
        return int(action.split('agent-'+str(receiver_id)+'_v-')[-1][0])

    else:
        return null_message_val
    # need to add values to these. # and make it easy to parse who it is going to . 
    # have it say agent-1, agent-2 etc. 


def initStatesandAgents():
    
    # need to randomize to avoid honest players learning which is Byzantine
    # index of Byzantines
    byzantine_inds = np.random.choice(range(num_agents), size=num_byzantine ,replace=False )

    #create the agents
    byzantine_list = []
    honest_list = []
    
    for i in range(num_agents):
        if i in byzantine_inds:
            byzantine_list.append(Agent(True, i, byzantine_inds))
        else: 
            honest_list.append(Agent(False, i))

    agent_list = byzantine_list + honest_list

    return agent_list, honest_list, byzantine_list


def giveReward(honest_parties):
    # checks to see if the honest parties have obtained both
    # consistency and validity 
    # returns honest then byzantine reward. 
    com_values = []
    starting_values = []
    for h in honest_parties:
        com_values.append(h.committed_value)
        starting_values.append(h.initVal)

    #checking if all the same value
    if len(set(com_values)) !=1:
        return (-1, 1)

    # checking validity
    if len(set(starting_values)) ==1:
        # if they are all the same and they havent 
        # agreed on the same value, then return -1
        if starting_values != com_values:
            return (-1, 1)

    return (1, -1)

def honestPartiesCommit(honest_list):
    for h in honest_list:
        if not h.committed_value:
            return False
    return True