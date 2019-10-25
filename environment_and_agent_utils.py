import numpy as np 
import torch
from config import *
from nn import *
import matplotlib.pyplot as plt

def toOneHotState(state):
    oh = []
    #print('the state to be made one hot', state)
    for s in state:
        #print('state for one hot',s)
        oh.append(oneHotStateMapper[s, :])
    oh = np.asarray(oh).flatten().T # now each column is one of the states.
    #convert ot pytorch tensor: 
    #print('the resulting one hot', oh)
    oh = torch.from_numpy(oh).float().to(device)
    return oh

def toOneHotActions(isByz, action_ind):
    # convert the action into an action ind:
    if isByz: 
        oh = byz_oneHotActionMapper[action_ind, :]
    else: 
        oh = honest_oneHotActionMapper[action_ind, :]
    oh = torch.from_numpy(oh).float().to(device)
    return oh 

class Agent:
    def __init__(self, isByzantine, agentID, byzantine_inds=None, give_inits = 0):
        self.isByzantine = isByzantine
        self.agentID = agentID
        if isByzantine:
            #self.brain = randomActions
            self.brain = byz_policy
        else: 
            #self.brain = randomActions
            self.brain = honest_policy
        self.actionSpace = getActionSpace(isByzantine, byzantine_inds, can_send_either_value=honest_can_send_either_value) 
        if type(give_inits) is not int:
            init_val = give_inits[agentID]
        else:
            init_val = np.random.choice(commit_vals, 1)[0]
        self.initVal = init_val
        initState = [init_val]
        if type(give_inits) is not int:
            #print('give', give_inits)
            for a in range(num_agents):
                if a == agentID:
                    continue
                else: 
                    initState.append(give_inits[a])
        else:
            for a in range(num_agents-1):
                initState.append(null_message_val)
        self.initState = initState
        self.state = self.initState
        self.committed_value = False

    def chooseAction(self, temperature, forceCommit=False):
   
        # look at the current state and decide what action to take. 
        oh = toOneHotState(self.state) # each column is one of the states. 
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

        return self.action, action_logprob
            
def updateStates(agent_list):
    #look at all agent actions and update the state of each to accomodate actions
    #print('agent list', agent_list)
    for reciever in agent_list:
        #print('reciever state going into the update', reciever.state)
        new_state = [reciever.initVal] # keep track of the agent's initial value, 
        #want to show to the NN each time
        actor_ind = 1 # this is always relative to the reciever. 
        for actor in agent_list:
            if actor == reciever: # check if agent committed. 
                continue
            # dont need to check if committed a value as already will prevent 
            # from taking any actions other than no send.
            new_state.append( actionEffect(actor.action, actor.initVal, reciever.state[actor_ind], reciever.agentID) )
            actor_ind +=1
        reciever.state = new_state

def actionEffect(action, init_val, actor_prev_action_result, receiver_id):
    # return the effects of a particular action

    if action == 'no_send':
        return null_message_val

    elif 'commit' in action: # keep returning the last state that the agent sent
        return actor_prev_action_result

    elif 'to_all' in action:
        if honest_can_send_either_value==False:
            if 'init' in action: # will only be true if honests not allowed to send a different value
                return init_val
        else: 
            return int(action.split('_')[-1])
    
    elif 'agent-'+str(receiver_id) in action:
        # getting the first value of this. 
        return int(action.split('agent-'+str(receiver_id)+'_v-')[-1][0])

    else:
        return null_message_val

def savePlot(to_plot, label):
    plt.figure()
    plt.plot(range(len(to_plot)), to_plot, label=label)
    plt.xlabel('epochs')
    plt.ylabel(label)
    plt.title(str(iters_per_epoch)+' iters per epoch')
    plt.legend()
    plt.gcf().savefig(directory+label+experiment_name+'.png', dpi=200)

def initStatesandAgents():
    
    # need to randomize to avoid honest players learning which is Byzantine
    # index of Byzantines
    byzantine_inds = np.random.choice(range(num_agents), size=num_byzantine ,replace=False )

    #create the agents
    byzantine_list = []
    honest_list = []

    #give_inits = list(np.random.choice([0,1], 3))
    #print(give_inits)
    #print(type(give_inits))
    
    for i in range(num_agents):
        if i in byzantine_inds:
            byzantine_list.append(Agent(True, i, byzantine_inds))
        else: 
            honest_list.append(Agent(False, i, )) #give_inits=give_inits

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
    '''no_send_true_value_first_penalty=np.array([0,0]) # may be able to remove this later. 
    #default if they did nothing wrong. 
    for key, trajectory_rounds in trajectory.items(): # going through the keys and their list of state, action, action prob pairs
        if 'Byz-False' in key: #only getting honest
            if honest_can_send_either_value==False: # make the string be init rather than a value. 
                string_to_check = 'init'
            else: 
                string_to_check = str(trajectory_rounds[0][1][0])
            if 'send_to_all-value_'+string_to_check != trajectory_rounds[0][2]: # getting the action from the first round
                return dishonesty_violation + round_penalty_total, satisfied_constraints
                #no_send_true_value_first_penalty=dont_send_value_first_round_penalty
                #break'''

    com_values = []
    starting_values = []
    for h in honest_parties:
        com_values.append(h.committed_value)
        starting_values.append(h.initVal)

    #checking if all the same value
    if len(set(com_values)) !=1:
        return consistency_violation + round_penalty_total, satisfied_constraints

    # want them to commit to the majority init value: 
    majority_init_value = np.floor((sum(starting_values)/len(starting_values))+0.5)
    if com_values[0] != majority_init_value: # as already made sure they were all the same value. 
        return majority_violation + round_penalty_total, satisfied_constraints

    # checking validity
    if len(set(starting_values)) ==1:
        # if they are all the same and they havent 
        # agreed on the same value, then return -1
        if starting_values != com_values:   
            return validity_violation + round_penalty_total, satisfied_constraints

    satisfied_constraints=True
    return correct_commit + round_penalty_total, satisfied_constraints

