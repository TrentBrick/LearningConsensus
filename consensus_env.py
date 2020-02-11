import numpy as np 
import torch
from nn import *
import matplotlib.pyplot as plt
import itertools
from collections import OrderedDict

# This file follows the OpenAI Gym environment API to interface with its RL algorithms

# need an actor critic with a .step also .reset and .render

# init the agents. 

# step function is one move forward for all of the agents. 
# their actions and decisions should then be batched up. need a conditional to deal with the byzantine. 
### returns a list of actions etc etc. 

# just implement it for the honest fisrt and then go from there. It is going to be harder in delineating each agent and what they are doing
# for byz and honest will need to modify the PPO code such that it loops through updating each of them. 
# the alternative is to ahve the bugger exist with each individual agent and perform updates on those. 
# this should be almost as fast as it otherwise would be. I can even average the buffer results from each of them. 

def onehotter(x, vocab_size):
    z = torch.zeros((x.shape[0], vocab_size))
    z.scatter_(1,x.flatten().long().unsqueeze(1),1)
    return z.view(x.shape[0], vocab_size)

class Agent:
    def __init__(self, params, pi, isByzantine, agentID, byzantine_inds=None, give_inits = 0,
    give_only_own_init=''):
        self.isByzantine = isByzantine
        self.agentID = agentID
        self.brain = pi
        self.actionSpace = getActionSpace(params, isByzantine, byzantine_inds, can_send_either_value=params['honest_can_send_either_value']) 
        self.actionDims = len(self.actionSpace)
        self.stateDims = params['num_agents'] * 3 # as the values are null value or one of the commits. 
       
        if isByzantine:
            init_val = params['null_message_val'] # doesnt need an init value 
        elif type(give_inits) is not int:
            init_val = give_inits[agentID]
        elif type(give_only_own_init) is not str: 
            init_val = give_only_own_init
        else:
            init_val = np.random.choice(params['commit_vals'], 1)[0]
        self.initVal = init_val
        initState = [init_val]
        if type(give_inits) is not int: # otherwise it is a list of the other values of everyone
            for a in range(params['num_agents']):
                if a == agentID:
                    continue
                else:
                    initState.append(give_inits[a])
                if params['use_PKI']: # need to set null values as nobody has recieved values from anybody else
                    initState += [params['null_message_val']]*(params['num_agents']-1)
        else:
            for a in range(params['num_agents']-1):
                initState.append(params['null_message_val'])
                if params['use_PKI']: # need to set null values as nobody has recieved values from anybody else
                    initState += [params['null_message_val']]*(params['num_agents']-1)
        self.initState = torch.tensor(initState).float()
        self.state = self.initState
        self.committed_value = False

    def chooseAction(self, oneHotStateMapper, temperature, device, forceCommit=False):
        # look at the current state and decide what action to take. 
        oh = onehotter(self.state, self.stateDims).flatten() # each column is one of the states. 
        #making a decision:
        logits = self.brain(oh)
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

        return self.action, action_logprob, action_ind.numpy()[0]
            
def updateStates(params, agent_list):

    # all actions are placed. agent sends what they recieved in the last round along withteir current messages. 

    # this internally resolves all of the agents and sets their new state. 
    # Now if PKI I need to add on to each state the states the other agents recieved
    if params['use_PKI']: # ordering looks like: [init val, agent 1 val, agent 2 val ..., tuple of their previous values]
        list_of_new_states = [] # need to get new states for all before updating them
        for reciever in agent_list:
            new_state = [reciever.initVal] # get the current state that has been set. 
            actor_ind = 1 # relative indexing. 
            for actor in agent_list:
                if actor == reciever: # check if agent committed. 
                    continue
                
                actor_action_consequence = actionEffect(params, actor.action, actor.initVal, reciever.state[actor_ind], reciever.agentID)
                new_state.append(actor_action_consequence) # appending what we already have
                new_state += actor.state[1:(len(agent_list)-1)] # getting the states from the previous period. 
                actor_ind +=1 # relative indexing. 
            list_of_new_states.append(new_state)
        
        for i, reciever in enumerate(agent_list): # actually update the states of each
            reciever.state = list_of_new_states[i] 

    else: 
        #look at all agent actions and update the state of each to accomodate actions
        #print('agent list', agent_list)
        for reciever in agent_list:
            #print('reciever state going into the update', reciever.state)
            new_state = [reciever.initVal] # keep track of the agent's initial value, 
            #want to show to the NN each time
            actor_ind = 1 # this is always relative to the reciever. We want indexes 1 and then 2 from their state space
            for actor in agent_list:
                if actor == reciever: # check if agent committed. 
                    continue
                # dont need to check if committed a value as already will prevent 
                # from taking any actions other than no send.
                new_state.append( actionEffect(params, actor.action, actor.initVal, reciever.state[actor_ind], reciever.agentID) )
                actor_ind +=1
            reciever.state = new_state
                

def actionEffect(params, action, init_val, actor_prev_action_result, receiver_id):
    # return the effects of a particular action

    if action == 'no_send':
        return params['null_message_val']

    elif 'commit' in action: # keep returning the last state that the agent sent
        return actor_prev_action_result

    elif 'to_all' in action:
        if params['honest_can_send_either_value']==False:
            if 'init' in action: # will only be true if honests not allowed to send a different value
                return init_val
        else: 
            return int(action.split('_')[-1])
    
    elif 'agent-'+str(receiver_id) in action:
        # getting the first value of this. 
        return int(action.split('agent-'+str(receiver_id)+'_v-')[-1][0])

    else:
        return params['null_message_val']


def getActionSpace(params, isByzantine, byzantine_inds=None, can_send_either_value=True):
    '''
    Creates a list of strings for the different actions that can be taken. 
    This provides not only the dimensions of the action space but also a way to 
    print the actions that have been taken. 
    '''

    if params['scenario'] == 'Basic':

        action_space = []

        if isByzantine:
            action_space.append('no_send')
            # no point in sending messages to other Byzantines as the central agent knows what the states are
            # but do have v granular send options.
            # and no commit option
            # get every possible combination of sending actions possible

            # remove the byz agents.
            non_byzantines = list(range(0, params['num_agents']))
            #print('byzantine inds', byzantine_inds)
            #print(non_byzantines)
            for byzantine_ind in byzantine_inds:
            	if byzantine_ind in non_byzantines:
                	non_byzantines.remove(byzantine_ind)
            #print('non byz are', non_byzantines)
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
                commit_val_permutes = list(itertools.permutations(params['commit_vals']*((choose_n//2)+1)))
                for combo_el in itertools.combinations(non_byzantines, choose_n):
                    for cvp in commit_val_permutes:
                        string = 'send'
                        for ind in range(choose_n):
                            string += '_agent-'+str(combo_el[ind])+'_v-'+str(cvp[ind])
                            #print('string', string)
                        action_space.append( string )
            # remove any redundancies in a way that preserves order.
            action_space = list(OrderedDict.fromkeys(action_space))

        else:
            if can_send_either_value:
                for commit_val in params['commit_vals']:
                    action_space.append('send_to_all-value_'+str(commit_val))
                    action_space.append('commit_'+str(commit_val))
            else:
                action_space.append('send_to_all-value_init')
                for commit_val in params['commit_vals']:
                    action_space.append('commit_'+str(commit_val))

    return action_space


if __name__=='__main__':
    print(torch.tensor([5,4,3,2]))
    # testing the onehotter. 
    ugh = torch.randn((5,10))
    print(ugh.argmax(-1))
    print(onehotter(ugh.argmax(-1), 10))
    print(onehotter(ugh.argmax(-1), 10).flatten())


