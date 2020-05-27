import numpy as np 
import torch
#from config import *
from nn import *
import matplotlib.pyplot as plt
import itertools
from collections import OrderedDict

def getActionSpace(params, isByzantine, byzantine_inds=None, can_send_either_value=True):
    '''move this to a new script that config and environment and agent utils can reference. '''

    # making the action space.
    # this depends on if the agent is Byzantine or not.
    ## if byzantine then global view over all Byzantines,
    # means that the same agent gets to act multiple times in the round
    # for honest, they only currently have option to send to everyone.
    #parties = set(range(params['num_agents']))
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

def toOneHotState(state, oneHotStateMapper, device):
    oh = []
    #print('the state to be made one hot', state)
    for s in state:
        #print('state for one hot',s)
        oh.append(oneHotStateMapper[s, :])
    oh = np.asarray(oh).flatten().T # now each column is one of the states.
    #print("one hot", oh, oh.shape)
    #convert ot pytorch tensor: 
    #print('the resulting one hot', oh)
    oh = torch.from_numpy(oh).float().to(device)
    return oh

def toOneHotActions(isByz, action_ind, byz_oneHotActionMapper, honest_oneHotActionMapper, device):
    # convert the action into an action ind:
    if isByz: 
        oh = byz_oneHotActionMapper[action_ind, :]
    else: 
        oh = honest_oneHotActionMapper[action_ind, :]
    oh = torch.from_numpy(oh).float().to(device)
    return oh 

class Agent:
    def __init__(self, params, honest_policy, byz_policy, isByzantine, agentID, byzantine_inds=None, give_inits = 0,
    give_only_own_init=''):
        self.isByzantine = isByzantine
        self.agentID = agentID
        if isByzantine:
            #self.brain = randomActions
            self.brain = byz_policy
        else: 
            #self.brain = randomActions
            self.brain = honest_policy
        self.actionSpace = getActionSpace(params, isByzantine, byzantine_inds, can_send_either_value=params['honest_can_send_either_value']) 
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
            #print('give', give_inits)
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
        self.initState = initState
        #print(' init state is: ', initState)
        self.state = self.initState
        self.committed_value = False

    def chooseAction(self, oneHotStateMapper, temperature, device, forceCommit=False):
        # look at the current state and decide what action to take. 
        oh = toOneHotState(self.state, oneHotStateMapper, device) # each column is one of the states. 
        #making a decision:
        #self.action = self.brain(self.actionSpace)
        # expects softmax to be applied to network first.
        logits = self.brain(oh)

        if forceCommit:
            if not self.isByzantine: #if it is an honest agent
                commit_inds = [ ind for ind, a in enumerate(self.actionSpace) if 'commit' in a ]
                logits = logits[commit_inds] # make it so that the only logits are those for committing. 
        #print('here are the logits after', logits)
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

def savePlot(params, to_plot, label, exp_dir):
    plt.figure()
    plt.plot(range(len(to_plot)), to_plot, label=label)
    plt.xlabel('epochs')
    plt.ylabel(label)
    plt.title(str(params['iters_per_epoch'])+' iters per epoch')
    plt.legend()
    plt.gcf().savefig(exp_dir+label+'.png', dpi=200)

def initStatesandAgents(params, honest_policy, byz_policy):
    
    # need to randomize to avoid honest players learning which is Byzantine
    # index of Byzantines
    byzantine_inds = np.random.choice(range(params['num_agents']), size=params['num_byzantine'] ,replace=False )

    #create the agents
    byzantine_list = []
    honest_list = []

    # giving the honest init vals to the byzantine. need to decide all of them here. 
    give_inits = list(np.random.choice([0,1], params['num_agents']))

    #print(give_inits)
    #print(type(give_inits))
    agent_list = []
    for i in range(params['num_agents']):
        if i in byzantine_inds:
            a = Agent(params, honest_policy, byz_policy, True, i, byzantine_inds, give_inits=give_inits)
            byzantine_list.append(a)
            agent_list.append(a)
        else: 
            a = Agent(params, honest_policy, byz_policy, False, i, give_only_own_init=give_inits[i])
            #print('honest', i, 'give init', type(give_inits[i]))
            honest_list.append(a) #give_inits=give_inits
            agent_list.append(a)

    return agent_list, honest_list, byzantine_list


def honestPartiesCommit(honest_list):
    for h in honest_list:
        if type(h.committed_value) is bool:
            return False
    return True

def giveReward(params, honest_parties, trajectory):
    # checks to see if the honest parties have obtained both
    # consistency and validity 
    # returns honest then byzantine reward. 

    satisfied_constraints = False
    #penalty for the number of rounds
    #num_rounds = len(trajectory[list(trajectory.keys())[0]]) # already iterating through the 

    com_values = []
    starting_values = []
    for h in honest_parties:
        com_values.append(h.committed_value)
        starting_values.append(h.initVal)

    #print(com_values)
    #print(starting_values)
    #print(trajectory)
    #checking if all the same value
    if len(set(com_values)) !=1:
        return params['consistency_violation'], satisfied_constraints

    # checking validity
    if len(set(starting_values)) ==1:
        # if they are all the same and they havent 
        # agreed on the same value, then return -1
        if starting_values != com_values:   
            return params['validity_violation'], satisfied_constraints

    # want them to commit to the majority init value: 
    if params['num_byzantine']==0:
        majority_init_value = np.floor((sum(starting_values)/len(starting_values))+0.5)
        if com_values[0] != int(majority_init_value): # as already made sure they were all the same value. 
            return params['majority_violation'], satisfied_constraints

    satisfied_constraints=True
    return params['correct_commit'], satisfied_constraints

