import numpy as np 
from consensus_env import getActionSpace, actionEffect

import torch
    


class Honest_Agent:

    def __init__(self, params, agentID, give_inits):
        self.isByzantine = False
        self.agentId = agentID 
        self.actionSpace = self.getHonestActionSpace(params)
        self.actionDims = len(self.actionSpace)
        self.stateDims = len(params['commit_vals'])+1 # +1 for the null value. 
        self.committed_ptr =  False
        self.reward = 0

        self.majority_value = None

        self.initVal = give_inits[agentID]
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
        for commit_val in params['commit_vals']:
            honest_action_space.append('send_to_all-new-value_'+str(commit_val))
            honest_action_space.append('commit_'+str(commit_val))
        return honest_action_space

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

    def step(self):
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
            new_state.append(actionEffect(self.params, actor.actionString, actor.initVal, agent.state[actor_ind], agent.agentId))
            actor_ind +=1
        agent.state = torch.tensor(new_state).int()








