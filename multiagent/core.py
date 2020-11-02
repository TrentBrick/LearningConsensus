import numpy as np 
from consensus_env import getActionSpace, actionEffect
import itertools
from collections import OrderedDict

import torch
    


class Honest_Agent:

    def __init__(self, params, agentId, give_inits):
        self.isByzantine = False
        self.agentId = agentId
        self.actionSpace = self.getHonestActionSpace(params)
        self.actionDims = len(self.actionSpace)
        self.stateDims = len(params['commit_vals'])+1 # +1 for the null value. 
        self.committed_ptr =  False
        self.reward = 0

        self.sentMajority = False

        self.majority_value = None

        self.initVal = give_inits[agentId]
        # self.initState = self.initAgentState(params, init_val, give_inits)
        #self.state = torch.tensor(self.initState).float()
        self.state = self.initAgentState(params, self.initVal, give_inits)
        self.committed_value = False

        self.actionIndex = None
        self.actionString = ""

        self.last_action_etc = dict()
        
        # can use this to create agents that don't react to the policy
        self.action_callback = None
    
    def initAgentState(self, params, init_val, give_inits):
        initState = [init_val]
        for a in range(params['num_agents']-1):
            initState.append(params['null_message_val'])
        return torch.tensor(initState).int()

    def getHonestActionSpace(self, params):
        # honest_action_space = getActionSpace(params, False, byzantine_inds=None, can_send_either_value=params['honest_can_send_either_value'])
        honest_action_space = []
        honest_action_space.append('send_to_all-value_init')
        # honest_action_space.append('sample_k_avalanche')
        for commit_val in params['commit_vals']:
            # honest_action_space.append('send_ll_'+str(commit_val))
            honest_action_space.append('commit_'+str(commit_val))
        return honest_action_space

class Byzantine_Agent:

    def __init__(self, params, agentId, give_inits, byzantine_inds):
        self.isByzantine = True
        self.agentId = agentId
        self.actionSpace = self.getByzantineActionSpace(params, byzantine_inds)
        self.actionDims = len(self.actionSpace)
        self.stateDims = len(params['commit_vals'])+1 # +1 for the null value. 
        self.committed_ptr =  False
        self.reward = 0

        self.sentMajority = False

        self.majority_value = None

        self.initVal = give_inits[agentId]
        # self.initState = self.initAgentState(params, init_val, give_inits)
        #self.state = torch.tensor(self.initState).float()
        self.state = self.initAgentState(params, self.initVal, give_inits)
        self.committed_value = False

        self.actionIndex = None
        self.action = ""

        self.last_action_etc = dict()
        # can use this to create agents that don't react to the policy
        self.action_callback = None


    def initAgentState(self, params, init_val, give_inits):
        initState = [init_val]
        for a in range(params['num_agents']-1):
            initState.append(params['null_message_val'])
        return torch.tensor(initState).int()

    def getByzantineActionSpace(self, params, byzantine_inds):
        action_space = []
        action_space.append('no_send')
         # remove the byz agents.
        non_byzantines = list(range(0, params['num_agents']))
        for byzantine_ind in byzantine_inds:
            if byzantine_ind in non_byzantines:
                non_byzantines.remove(byzantine_ind)

        # Get all combinations of the honest agents to send to
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

        # Only give option to send to two agents 
        action_space = action_space[5:]

        return action_space
        
    
#multi-agent world
class World(object):
    
    def __init__(self, params):
        self.params = params
        self.agents = []
        ##TODO: might need to add some dimensions for action spaces here
    
    #return all agents controlled by a policy 
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self, curr_sim_len):
        for agent in self.agents:
            self.update_agent_state(agent, self.agents)
            agent.majorityValue = np.floor((sum(agent.state)/len(agent.state))+0.5)


    def update_agent_state(self, agent, agent_list):
        #Update the given agents state given what all the other agents actions are 
        actor_ind = 1
        new_state = [agent.initVal] # keep track of the agent's initial value
        for actor in agent_list:
            if agent.agentId == actor.agentId:
                continue
            new_state.append(actionEffect(self.params, self.agents, actor.actionString, actor.initVal, agent.state[actor_ind], agent.agentId))
            actor_ind +=1
        agent.state = torch.tensor(new_state).int()








