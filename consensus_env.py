import numpy as np 
import torch
from nn import *
import matplotlib.pyplot as plt
import spinup.algos.pytorch.ppo.core as core
import itertools
from collections import OrderedDict
#from actions import getActionSpace, actionEffect

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

def getHonestActionSpace(params):
    if params['scenario']=='Basic':
        honest_action_space = getActionSpace(params, False, byzantine_inds=None, can_send_either_value=params['honest_can_send_either_value'])
        honest_action_space_size = len(honest_action_space)
        #honest_action_to_ind = {a:ind for ind, a in enumerate(honest_action_space)}
    return honest_action_space, honest_action_space_size#, honest_action_to_ind

def getByzantineActionSpace(params):
    if params['scenario']=='Basic':  
        byz_action_space = getActionSpace(params, True, byzantine_inds=[0], can_send_either_value=params['honest_can_send_either_value'])
        byz_action_space_size = len(byz_action_space)
        #byz_action_to_ind = {a:ind for ind, a in enumerate(byz_action_space)}
        #print('byz action to ind', )
    return byz_action_space, byz_action_space_size#, byz_action_to_ind

def actionEffect(params, actionStr, init_val, actor_prev_action_result, receiver_id):
    # return the effects of a particular action

    if actionStr == 'no_send':
        return params['null_message_val']

    elif 'commit' in actionStr: # keep returning the last state that the agent sent
        return actor_prev_action_result

    elif actionStr == 'send_to_all-value_init':
        return init_val

    elif 'send_to_all-value_' in actionStr:
        return int(actionStr.split('_')[-1])
    
    elif 'agent-'+str(receiver_id) in actionStr:
        # getting the first value of this. 
        return int(actionStr.split('agent-'+str(receiver_id)+'_v-')[-1][0])

    else:
        return params['null_message_val']

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
    def __init__(self, params, neural_net, value_function, isByzantine, agentID, byzantine_inds=None, give_inits = 0,
    give_only_own_init=''):
        self.isByzantine = isByzantine
        self.agentID = agentID
        self.brain = neural_net
        self.value_function = value_function
        self.actionSpace = getActionSpace(params, isByzantine, byzantine_inds) 
        self.actionDims = len(self.actionSpace)

        self.stateDims = len(params['commit_vals'])+1 # +1 for the null value. 

        '''if params['use_PKI']: 
            self.stateDims = (len(params['commit_vals'])+1)*(params['num_agents']**2-params['num_agents']+1)
        else: 
            self.stateDims = (len(params['commit_vals'])+1)*params['num_agents']
        '''
        local_rounds_per_epoch = params['rounds_per_epoch'] // params['ncores']
        self.buffer = PPOBuffer(self.stateDims, self.actionDims, local_rounds_per_epoch, gamma=params['gamma'], lam=params['lam'])

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

    def chooseAction(self, oneHotStateMapper, temperature, forceCommit=False): # device,
        # look at the current state and decide what action to take. 
        oh = onehotter(self.state, self.stateDims).flatten() # each column is one of the states. 
        #making a decision:
        logits = self.brain(oh)
        real_logprobs = torch.log(torch.nn.functional.softmax(logits, dim=0)) # currently not vectorized
        #should be able to apply sampling without computing this twice... 
        temperature_probs =  torch.nn.functional.softmax(logits/temperature, dim=0) 
        action_ind = torch.multinomial(temperature_probs, 1, replacement=False) # returns 1 sample per row. 
        action_logprob = real_logprobs[action_ind]
        self.action = action_ind
        self.actionStr = self.actionSpace[action_ind]

        ###If commit in agents action space, then commit
        if 'commit' in self.actionStr:
            self.committed_value = int(self.action.split('_')[1])

        value = self.value_function(oh)
            
        return self.action, action_logprob, value
            
def updateStates(params, agent_list, honest_list):

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
                
                actor_action_consequence = actionEffect(params, actor.actionStr, actor.initVal, reciever.state[actor_ind], reciever.agentID)
                new_state.append(actor_action_consequence) # appending what we already have
                new_state += actor.state[1:(len(agent_list)-1)] # getting the states from the previous period. 
                actor_ind +=1 # relative indexing. 
            list_of_new_states.append(new_state)
        
        for i, reciever in enumerate(agent_list): # actually update the states of each
            reciever.state = list_of_new_states[i] 

    else: 
        #look at all agent actions and update the state of each to accomodate actions
        
        for reciever in agent_list:
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

    # calculate the rewards: 
    rewards, sim_done = giveRewards(params, agent_list, honest_list)

    return reward, sim_done

def giveRewards(params, agent_list, honest_list):
    # checks to see if the honest parties have obtained both
    # consistency and validity in addition to any other rewards

    rewards = np.zeros((len(agent_list)))
    sim_done = False
    # first check if all honest have committed and give the final reward if so
    all_committed = True
    comm_values = []
    starting_values = [] # needed to check validity
    for h in honest_list: 
        if h.committed_value is bool:
            all_committed = False
            break
        else: 
            comm_values.append(h.committed_value)
            starting_values.append(h.initVal)
    
    if all_committed: 
        sim_done = True
        honest_comm_reward, satisfied_constraints = getCommReward(params, comm_values, starting_values)
        for i, a in enumerate(agent_list):
            if not a.isByz:
                rewards[i] += honest_comm_reward
            else: 
                rewards[i] -= honest_comm_reward

        return rewards, sim_done

    for i, a in enumerate(agent_list): 
        # reward for sending to all in the first round
        if a.isByz == False and a.buffer.ptr == 0 and 'send_to_all-' in a.action:
            rewards[i] += params['send_all_first_round_reward']

        # round length penalties
        if not a.isByz:
            rewards_to_go += params['additional_round_penalty']
        elif a.isByz: 
            rewards_to_go -= params['additional_round_penalty']

    return rewards, sim_done # NEED TO DISTINGUISH BETWEEN AGENT BEING DONE AND A WHOLE ROUND BEING DONE. 

def getCommReward(params, comm_values, starting_values):

    satisfied_constraints = False

    if len(set(comm_values)) !=1:
        return params['consistency_violation'], satisfied_constraints

    # checking validity
    if len(set(starting_values)) ==1:
        # if they are all the same and they havent 
        # agreed on the same value, then return -1
        if starting_values != comm_values:   
            return params['validity_violation'], satisfied_constraints

    # want them to commit to the majority init value: 
    if params['num_byzantine']==0:
        majority_init_value = np.floor((sum(starting_values)/len(starting_values))+0.5)
        if comm_values[0] != int(majority_init_value): # as already made sure they were all the same value. 
            return params['majority_violation'], satisfied_constraints

    satisfied_constraints=True
    return params['correct_commit'], satisfied_constraints

class ConsensusEnv():
    def __init__(self, params):
        self.params = params 

        honest_action_space, honest_action_space_size = getHonestActionSpace(params)
        byz_action_space, byz_action_space_size = getByzantineActionSpace(params)
        
        #Get state_oh_size
        if params['use_PKI']: 
            state_oh_size = (len(params['commit_vals'])+1)*(params['num_agents']**2-params['num_agents']+1)
        else: 
            state_oh_size = (len(params['commit_vals'])+1)*params['num_agents']

        print('neural net inits', honest_action_space_size, state_oh_size)
        print('nerual net ativation', params['activation'])
        self.honest_policy = BasicPolicy(honest_action_space_size, state_oh_size, params['hidden_sizes'], params['activation'], params['output_activation'], params['use_bias'])#.to(params['device'])
        self.byz_policy = BasicPolicy(byz_action_space_size, state_oh_size, params['hidden_sizes'], params['activation'], params['output_activation'], params['use_bias'])#.to(params['device'])

        self.honest_optimizer = torch.optim.Adam(self.honest_policy.parameters(), lr=params['learning_rate'])
        self.byz_optimizer = torch.optim.Adam(self.byz_policy.parameters(), lr=params['learning_rate'])

        self.oneHotStateMapper = np.eye(len(params['commit_vals'])+1) # number of unique values that can be in the state. 
        self.honest_oneHotActionMapper = np.eye(honest_action_space_size)
        self.byz_oneHotActionMapper = np.eye(byz_action_space_size)
        
        #TODO: Need to call these from params
        adv_hidden_sizes = (16,8)
        adv_learning_rate=0.003
        adv_activation= torch.relu
        adv_output_activation = None # I do softmax in the env section.
        adv_use_bias = True
        adv_output_size = 1
        # currently byz and honest use the same network sizes and learning rates.
        #TODO: shouldn't the value function take in the observation space?
        self.honest_v_function = BasicPolicy(adv_output_size, state_oh_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias)#.to(device)
        #honest_q_function = BasicPolicy(adv_output_size, state_oh_size+honest_action_space_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias).to(device)
        self.honest_v_function_optimizer = torch.optim.Adam(self.honest_v_function.parameters(), lr=adv_learning_rate)
        #honest_q_function_optimizer = torch.optim.Adam(honest_q_function.parameters(), lr=adv_learning_rate)

        self.byz_v_function = BasicPolicy(adv_output_size, state_oh_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias)#.to(device)
        #byz_q_function = BasicPolicy(adv_output_size, state_oh_size+byz_action_space_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias).to(device)
        self.byz_v_function_optimizer = torch.optim.Adam(self.byz_v_function.parameters(), lr=adv_learning_rate)
        #byz_q_function_optimizer = torch.optim.Adam(byz_q_function.parameters(), lr=adv_learning_rate)

        #adv_optimizers = [honest_v_function_optimizer, honest_q_function_optimizer, byz_v_function_optimizer, byz_q_function_optimizer]
        #adv_optimizers = [self.honest_v_function_optimizer, self.byz_v_function_optimizer]

        #for net in [honest_v_function, honest_q_function, byz_v_function, byz_q_function]:
        for net in [self.honest_v_function, self.byz_v_function]:
            net.train()
            net.zero_grad()

        #### Code for loading in a policy ####
        # if params['load_policy_honest'] != "None":
        # print("LOADING IN an honest policy, load_policy=True")
        # honest_policy = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_honest']+'.torch')
        # if params['use_vpg']:
        #     honest_v_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_honest']+'_v'+'.torch')
        #     honest_q_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_honest']+'_q'+'.torch')
        # if params['load_policy_byz'] != "None":
        #     byz_policy = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_byz']+'.torch')
        #     if params['use_vpg']: 
        #         byz_v_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_byz']+'_v'+'.torch')
        #         byz_q_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_byz']+'_q'+'.torch')
                #encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss, curr_ep, best_eval_acc = loadpolicy(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, load_name)

        ##############################################
        if params['train_honest']:
            self.honest_policy.train()
        else: 
            self.honest_policy.eval()
        if params['train_byz']:
            self.byz_policy.train() # the value functions i iterate through in main for train().
        else: 
            self.byz_policy.eval() 

        self.honest_policy.zero_grad()
        self.byz_policy.zero_grad()

        # TODO: if these are self then dont need to pass them into the class function. 
        self.agent_list, self.honest_list, self.byzantine_list = self.initStatesandAgents()

    def reset(self):
        self.agent_list, self.honest_list, self.byzantine_list = self.initStatesandAgents()

    def initStatesandAgents(self):
    
        # need to randomize to avoid honest players learning which is Byzantine
        # index of Byzantines
        byzantine_inds = np.random.choice(range(self.params['num_agents']), size=self.params['num_byzantine'] ,replace=False )

        #create the agents
        byzantine_list = []
        honest_list = []

        # giving the honest init vals to the byzantine. need to decide all of them here. 
        give_inits = list(np.random.choice([0,1], self.params['num_agents']))

        #print(give_inits)
        #print(type(give_inits))
        agent_list = []
        for i in range(self.params['num_agents']):
            if i in byzantine_inds:
                a = Agent(self.params, self.byz_policy, self.byz_v_function, True, i, byzantine_inds, give_inits=give_inits)
                byzantine_list.append(a)
                agent_list.append(a)
            else: 
                a = Agent(self.params, self.honest_policy, self.honest_v_function, False, i, give_only_own_init=give_inits[i])
                #print('honest', i, 'give init', type(give_inits[i]))
                honest_list.append(a) #give_inits=give_inits
                agent_list.append(a)
        return agent_list, honest_list, byzantine_list

    def step(self, ep_len, total_ep_rounds):
        # this step needs to iterate through all of the agents. it doesnt need to return
        # anything though as each agent has their own buffer. 

        # choose new actions: 
        for agent in self.agent_list: 
            if agent.isByzantine: 
                # TODO: implement temperature tracking
                curr_temperature = 1.0#byz_curr_temperature
            else: 
                curr_temperature = 1.0 #honest_curr_temperature

            # TODO: need to return a value along with the action taken!!! 
            if type(agent.committed_value) is int: # dont change to True! Either it is False or a real value. 
                a, logp, v = agent.action, None, None
            else:
                a, logp, v = agent.chooseAction(self.oneHotStateMapper, curr_temperature)
                
            # update the environment for each agent and calculate the reward here also if the simulation has terminated.  
            rewards, sim_done = updateStates(self.params, self.agent_list, self.honest_list)

            for ind, agent in enumerate(self.agent_list): 

                agent.buffer.store(o, a, rewards[ind], v, logp)

                ep_len += 1
                timeout = ep_len == self.params['max_round_len']
                terminal = sim_done or timeout
                epoch_ended = total_ep_rounds==agent.buffer.obs_buf.shape[0] -1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _,_, v = agent.chooseAction(oneHotStateMapper, curr_temperature) #, device)
                    else:
                        v = 0
                    agent.buffer.finish_path(v) # tie off this path no matter what
                    #TODO: fix the logger here. 
                    #if terminal:
                        # only save EpRet / EpLen if trajectory finished
                    #    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    #o, ep_ret, ep_len = env.reset(), 0, 0 # reset the environment
                    
                    #termination_list.append(1)

        '''end_sim =False
        if sum(termination_list) == num_agents or honestPartiesCommit(honest_list):
            end_sim=True'''

        return v, end_sim

        # store in the buffer. 
        
    
            
        # then store to the buffer. 


                # log the current state and action

    def render(self,  mode='human', close=False):
        print("This is a test of rendering the environment ")



class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    size: the number of rounds in the trajectory
    obs_dim: dimension of obs
    act_dim: actions
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew # the actual reward recieved. 
        self.val_buf[self.ptr] = val # the value function estimate. 
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val) # adding the very last value. 
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation.
        # this is much more sophisticated than the basic advantage equation. 
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr # making the new start point to add another new simulation in at the end!

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get. what if it ends early???
        
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0

if __name__=='__main__':
    print(torch.tensor([5,4,3,2]))
    # testing the onehotter. 
    ugh = torch.randn((5,10))
    print(ugh.argmax(-1))
    print(onehotter(ugh.argmax(-1), 10))
    print(onehotter(ugh.argmax(-1), 10).flatten())


