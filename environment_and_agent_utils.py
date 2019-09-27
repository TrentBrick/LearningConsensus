import numpy as np 
import itertools
from collections import OrderedDict
from config import *
from nn import *

def toOneHot(state):
    oh = []
    #print('full state for an agent', state)
    for s in state:
        #print('state for one hot',s)
        oh.append(oneHotMapper[s])
    return np.asarray(oh).T # now each column is one of the states. 

class Agent:
    def __init__(self, isByzantine, agentID, byzantine_inds=None):
        self.isByzantine = isByzantine
        self.agentID = agentID
        if isByzantine:
            self.brain = randomActions
            '''self.brain = byz_policy'''
        else: 
            self.brain = randomActions
            '''self.brain = honest_policy'''
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

    def chooseAction(self):
        if self.committed_value != False: # dont allow to take any actions. just keep committing.  
            self.action = 'commit_'+str(self.committed_value)
        else: 
            # look at the current state and decide what action to take. 
            oh = toOneHot(self.state) # each column is one of the states. 
            #making a decision:
            self.action = self.brain(self.actionSpace)
            '''probabilities = self.brain(oh)
            action_ind = torch.max(probabilities)
            self.action = self.actionSpace[action_ind]'''

            if self.action.split('_')[0]== 'commit': # checking for a commit. 
                self.commitValue(self.action.split('_')[1])

        return self.state, self.action
            
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


def getActionSpace(isByzantine, byzantine_inds=None):

    # making the action space. 
    # this depends on if the agent is Byzantine or not. 
    ## if byzantine then global view over all Byzantines, 
    # means that the same agent gets to act multiple times in the round
    # for honest, they only currently have option to send to everyone. 
    parties = set(range(num_agents))
    if scenario == 'Basic':

        action_space = ['no_send']

        if isByzantine:
            # no point in sending messages to other Byzantines as the central agent knows what the states are
            # but do have v granular send options. 
            # and no commit option
            # get every possible combination of sending actions possible
            
            # remove the honest agents. 
            non_byzantines = list(range(0, num_agents))
            for byzantine_ind in byzantine_inds:
                non_byzantines.remove(byzantine_ind)
            #for val in commit_vals:
            #    non_byzantines.append('v'+str(val)) # add in the possible values that can be sent

            # this code is tricky, I get all combinations of the honest agents to send to
            # and then interleave in all permutations of the values that can be sent to them. 
            # for example a subset of them is: : 
            ''''send_agent-2_value-0',
            'send_agent-1_value-0_agent-3_value-0',
            'send_agent-1_value-0_agent-2_value-1_agent-3_value-1',
            'send_agent-1_value-1_agent-3_value-1',
            'send_agent-1_value-1_agent-2_value-1_agent-3_value-0',
            'send_agent-1_value-0_agent-2_value-0_agent-3_value-0',
            'send_agent-1_value-1_agent-2_value-1',
            'send_agent-2_value-1_agent-3_value-1','''
            for choose_n in range(1, len(non_byzantines)+1):
                commit_val_permutes = list(itertools.permutations(commit_vals*((choose_n//2)+1)))
                for combo_el in itertools.combinations(non_byzantines, choose_n):
                    for cvp in commit_val_permutes:
                        string = 'send'
                        for ind in range(choose_n):
                            string += '_agent-'+str(combo_el[ind])+'_v-'+str(cvp[ind])
                        action_space.append( string ) 
            # remove any redundancies in a way that preserves order. 
            action_space = list(OrderedDict.fromkeys(action_space))

        else:
            for commit_val in commit_vals:
                action_space.append('send_to_all-value_'+str(commit_val)) 
                action_space.append('commit_'+str(commit_val)) 

    return action_space

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

def getEnv(scenario, num_agents, commit_vals):

    pass

def honestPartiesCommit(honest_list):
    for h in honest_list:
        if not h.committed_value:
            return False
    return True