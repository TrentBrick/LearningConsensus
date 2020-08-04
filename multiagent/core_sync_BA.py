import numpy as np 
from consensus_env import getActionSpace, actionEffect, sync_BA_effect  
import itertools
from collections import OrderedDict

import torch
    


class Honest_Agent:

    def __init__(self, params, agentId, give_inits):
        self.isLeader = False
        self.isByzantine = False
        self.agentId = agentId
        self.stateDims = len(params['commit_vals'])+1 # +1 for the null value. 
        self.committed_ptr =  False
        self.reward = 0

        self.state = self.initAgentState(params, give_inits)
        self.committed_value = -1

        self.actionIndex = None
        self.actionString = ""

        self.last_action_etc = dict()
        # can use this to create agents that don't react to the policy
        self.action_callback = True
        self.isLeader = False
        self.proposeValue = params['null_message_val']
        self.statusValue = params['null_message_val']
    
    def initAgentState(self, params, give_inits):
        initState = []
        for a in range(params['num_agents']):
            initState.append(params['null_message_val'])
        return torch.tensor(initState).int()


class Byzantine_Agent:

    def __init__(self, params, agentId, give_inits, byzantine_inds):
        self.isByzantine = True
        self.agentId = agentId
        self.actionSpace = self.getByzantineActionSpace(params, byzantine_inds)
        self.actionDims = len(self.actionSpace)
        self.stateDims = len(params['commit_vals'])+1 # +1 for the null value. 
        self.committed_ptr =  False
        self.reward = 0

        self.proposeValue = params['null_message_val']
        self.statusValue = params['null_message_val']


        self.committed_value = False

        self.actionIndex = None
        self.actionString = ""
        self.prevActionString = ""

        self.last_action_etc = dict()
        # can use this to create agents that don't react to the policy
        self.action_callback = None
        self.isLeader = True

        state = self.initAgentState(params, give_inits)
        if self.isLeader:
            state.append(1)
        else:
            state.append(0)

        #No send
        # 2 actions where send something different
        state = (state + [0]*5)
        state.append(1)
        state.append(0)
        state.append(1)
        state.append(0)
        # state = (state + [0]*6)
        self.state = torch.tensor(state).int()
        # print(self.state)
    




    def initAgentState(self, params, give_inits):
        initState = []
        for a in range(params['num_agents']):
            initState.append(params['null_message_val'])
        
        ## Append for round
        initState.append(1)

        #Leader can send anything in the first round
        return initState

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
        # print(action_space)
        # print(action_space)
        # Only give option to send to two agents 
        # action_space = action_space[1:5]

        # rounds = ['propose_', 'vote_', 'status_']
        # prefix_action_space = []
        # for prefix in rounds:
        #     for action in action_space:
        #         prefix_action_space.append(prefix+action)
        # print(action_space)
        # action_space = action_space[5:]

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
        if (curr_sim_len < 3):
            for agent in self.agents:
                self.update_agent_state(agent, self.agents, curr_sim_len)
        
        ## Based on the actions of the other two honest agents, limit the action space
        # if curr_sim_len is 1:
            ## Can't send different value from what it sent previously 

                

    def update_agent_state(self, agent, agent_list, curr_sim_len):
        #Update the given agents state given what all the other agents actions are 
        actor_ind = 1
        new_state = [agent.proposeValue] # keep track of the agent's initial value
        for actor in agent_list:
            if agent.agentId == actor.agentId:
                continue
            new_state.append(sync_BA_effect(self.params, self.agents, actor.actionString, agent.state[actor_ind], agent.agentId, curr_sim_len))
            actor_ind +=1
        
        #Update initial value - can't override this value later when the byzantine sends it 
        if curr_sim_len == 1 and not agent.isLeader:
            for val in new_state:
                if val != self.params['null_message_val']:
                    new_state[0] = val
                    agent.proposeValue = val
        
        if agent.isByzantine:
            new_state.append(curr_sim_len)
            
            if agent.isLeader:
                new_state.append(1)
            else:
                new_state.append(0)

            new_state = (new_state + [0]*5)
            new_state.append(1)
            new_state.append(0)
            new_state.append(1)
            new_state.append(0)
            # print('sim is: ', curr_sim_len)
            # print(new_state)
            
        agent.state = torch.tensor(new_state).int()








