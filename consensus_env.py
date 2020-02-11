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
        
        commitDict = {}
        for ind, a in enumerate(self.actionSpace):
            if 'commit' in a: 
                commitDict[ind] = int(a.split('_')[1]) # gets the commit value associated with this. 
            else:
                commitDict[ind] = params['null_message_val']

        self.commitDict = commitDict

        self.actionDims = len(self.actionSpace)

        if params['use_PKI']: 
            self.stateDims = (len(params['commit_vals'])+1)*(params['num_agents']**2-params['num_agents']+1)
        else: 
            self.stateDims = (len(params['commit_vals'])+1)*params['num_agents']

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
        self.committed_value = params['null_message_val']

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
        self.action = action_ind
        self.actionStr = self.actionSpace[action_ind]
        self.committed_value = self.commitDict[action_ind]
        return self.action, action_logprob
            
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
                
                actor_action_consequence = actionEffect(params, actor.actionStr, actor.initVal, reciever.state[actor_ind], reciever.agentID)
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
                
def actionEffect(params, actionStr, init_val, actor_prev_action_result, receiver_id):
    # return the effects of a particular action

    if actionStr == 'no_send':
        return params['null_message_val']

    elif 'commit' in actionStr: # keep returning the last state that the agent sent
        return actor_prev_action_result

    elif 'to_all' in actionStr:
        if params['honest_can_send_either_value']==False:
            if 'init' in actionStr: # will only be true if honests not allowed to send a different value
                return init_val
        else: 
            return int(actionStr.split('_')[-1])
    
    elif 'agent-'+str(receiver_id) in actionStr:
        # getting the first value of this. 
        return int(actionStr.split('agent-'+str(receiver_id)+'_v-')[-1][0])

    else:
        return params['null_message_val']

class ConsensusEnv:
    def __init__(params):
        self.params = params 

        honest_action_space, honest_action_space_size = getHonestActionSpace(params)
        byz_action_space, byz_action_space_size = getByzantineActionSpace(params)
        
        self.honest_policy = BasicPolicy(honest_action_space_size, state_oh_size, params['hidden_sizes'], activation, output_activation, params['use_bias']).to(device)
        self.byz_policy = BasicPolicy(byz_action_space_size, state_oh_size, params['hidden_sizes'], activation, output_activation, params['use_bias']).to(device)

        honest_optimizer = torch.optim.Adam(honest_policy.parameters(), lr=params['learning_rate'])
        byz_optimizer = torch.optim.Adam(byz_policy.parameters(), lr=params['learning_rate'])

        oneHotStateMapper = np.eye(len(params['commit_vals'])+1) # number of unique values that can be in the state. 
        honest_oneHotActionMapper = np.eye(honest_action_space_size)
        byz_oneHotActionMapper = np.eye(byz_action_space_size)
        ## Initialize vpg
        if params['rl_algo_wanted']=='vpg' and params['use_vpg']:
            adv_hidden_sizes = (16,8)
            adv_learning_rate=0.003
            adv_activation= torch.relu
            adv_output_activation = None # I do softmax in the env section.
            adv_use_bias = True
            adv_output_size = 1
            # currently byz and honest use the same network sizes and learning rates.
            honest_v_function = BasicPolicy(adv_output_size, state_oh_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias).to(device)
            honest_q_function = BasicPolicy(adv_output_size, state_oh_size+honest_action_space_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias).to(device)
            honest_v_function_optimizer = torch.optim.Adam(honest_v_function.parameters(), lr=adv_learning_rate)
            honest_q_function_optimizer = torch.optim.Adam(honest_q_function.parameters(), lr=adv_learning_rate)

            byz_v_function = BasicPolicy(adv_output_size, state_oh_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias).to(device)
            byz_q_function = BasicPolicy(adv_output_size, state_oh_size+byz_action_space_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias).to(device)
            byz_v_function_optimizer = torch.optim.Adam(byz_v_function.parameters(), lr=adv_learning_rate)
            byz_q_function_optimizer = torch.optim.Adam(byz_q_function.parameters(), lr=adv_learning_rate)

            adv_optimizers = [honest_v_function_optimizer, honest_q_function_optimizer, byz_v_function_optimizer, byz_q_function_optimizer]

            for net in [honest_v_function, honest_q_function, byz_v_function, byz_q_function]:
                net.train()
                net.zero_grad()

        if params['load_policy_honest'] != "None":
        print("LOADING IN an honest policy, load_policy=True")
        honest_policy = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_honest']+'.torch')
        if params['use_vpg']:
            honest_v_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_honest']+'_v'+'.torch')
            honest_q_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_honest']+'_q'+'.torch')
        if params['load_policy_byz'] != "None":
            byz_policy = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_byz']+'.torch')
            if params['use_vpg']: 
                byz_v_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_byz']+'_v'+'.torch')
                byz_q_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_byz']+'_q'+'.torch')
                #encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss, curr_ep, best_eval_acc = loadpolicy(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, load_name)

        if params['train_honest']:
            honest_policy.train()
        else: 
            honest_policy.eval()
        if params['train_byz']:
            byz_policy.train() # the value functions i iterate through in main for train().
        else: 
            byz_policy.eval() 

        honest_policy.zero_grad()
        byz_policy.zero_grad()

        if params['rl_algo_wanted']=='vpg' and params['use_vpg']:
            for net in [honest_v_function, honest_q_function, byz_v_function, byz_q_function]:
                net.train()

        # TODO: if these are self then dont need to pass them into the class function. 
        self.agent_list, self.honest_list, self.byzantine_list = initStatesandAgents(self.params, self.honest_policy, self.byz_policy)

    def reset():
        self.agent_list, self.honest_list, self.byzantine_list = initStatesandAgents(self.params, self.honest_policy, self.byz_policy)


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

    def step(ep_len, total_ep_rounds):
        # this step needs to iterate through all of the agents. it doesnt need to return
        # anything though as each agent has their own buffer. 

        termination_list = []
            # choose new actions: 
            for agent in self.agent_list: 
                if agent.isByzantine: 
                    curr_temperature = byz_curr_temperature
                else: 
                    curr_temperature = honest_curr_temperature

                # TODO: need to return a value along with the action taken!!! 

                if type(agent.committed_value) is int: # dont change to True! Either it is False or a real value. 
                    a, logp, v = agent.action, None, None
                
                else:
                    a, logp, v = agent.chooseAction(oneHotStateMapper, curr_temperature, device)
                
                # update the environment for each agent and calculate the reward here. 
                r, d = updateState(self.params, self.agent_list)

                agent.buffer.store(o, a, r, v, logp)

                ep_len += 1
                timeout = ep_len == max_ep_len self.params['max_round_len']
                terminal = d or timeout
                epoch_ended = total_ep_rounds==agent.buffer.obs_buf.shape[0] -1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _,_, v = agent.chooseAction(oneHotStateMapper, curr_temperature, device)
                    else:
                        v = 0
                    agent.buffer.finish_path(v) # tie off this path no matter what
                    #TODO: fix the logger here. 
                    #if terminal:
                        # only save EpRet / EpLen if trajectory finished
                    #    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    #o, ep_ret, ep_len = env.reset(), 0, 0 # reset the environment
                    termination_list.append(1)

        end_sim =False
        if sum(termination_list) == num_agents or honestPartiesCommit(honest_list):
            end_sim=True

        return v, end_sim

        # store in the buffer. 
        
    
            
        # then store to the buffer. 


                # log the current state and action





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
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

if __name__=='__main__':
    print(torch.tensor([5,4,3,2]))
    # testing the onehotter. 
    ugh = torch.randn((5,10))
    print(ugh.argmax(-1))
    print(onehotter(ugh.argmax(-1), 10))
    print(onehotter(ugh.argmax(-1), 10).flatten())


