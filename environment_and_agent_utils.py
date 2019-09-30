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

    '''def commitValue(self, value):
        self.committed_value = value'''

    def chooseAction(self, temperature, forceCommit=False):
   
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

        if forceCommit: # if forcecommit, convert the chosen action back to commit and action space relevance. 
            if not self.isByzantine:
                self.action = self.actionSpace[ commit_inds[action_ind] ]

        if 'commit' in self.action: # checking for a commit. 
            self.committed_value = int(self.action.split('_')[1])

        return self.action, action_logprob
            
def updateStates(agent_list):
    #look at all agent actions and update the state of each to accomodate actions

    for reciever in agent_list:
        new_state = [reciever.initVal] # keep track of the agent's initial value, 
        #want to show to the NN each time
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

def honestPartiesCommit(honest_list):
    for h in honest_list:
        if type(h.committed_value) is bool:
            return False
    return True

def giveReward(honest_parties, trajectory):
    # checks to see if the honest parties have obtained both
    # consistency and validity 
    # returns honest then byzantine reward. 

    satisfied_constraints = False
    #penalty for the number of rounds
    num_rounds = len(trajectory[list(trajectory.keys())[0]]) # already iterating through the 
    round_penalty_total = num_rounds*round_penalty

    # if they started by committing, then punish: 
    # a more handholding version would be to encourage them to send the first round. 
    commit_penalty=np.array([0,0]) # default if they did nothing wrong. 
    for key, trajectory_rounds in trajectory.items(): # going through the keys and their list of state, action, action prob pairs
        if 'Byz-False' in key: #only getting honest
            if 'commit' in trajectory_rounds[0][2] or 'no_send' in trajectory_rounds[0][2]: # getting the action from the first round
                commit_penalty=commit_first_round_penalty
                break

    com_values = []
    starting_values = []
    for h in honest_parties:
        com_values.append(h.committed_value)
        starting_values.append(h.initVal)

    #checking if all the same value
    if len(set(com_values)) !=1:
        return consistency_violation + commit_penalty + round_penalty_total, satisfied_constraints

    # checking validity
    if len(set(starting_values)) ==1:
        # if they are all the same and they havent 
        # agreed on the same value, then return -1
        if starting_values != com_values:   
            return validity_violation + commit_penalty + round_penalty_total, satisfied_constraints

    satisfied_constraints=True
    return correct_commit + commit_penalty + round_penalty_total, satisfied_constraints

