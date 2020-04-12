import numpy as np 
from consensus_env import getActionSpace, actionEffect
import torch
    


class Honest_Agent:

    def __init__(self, params, neural_net, value_function, agentID, give_inits):
        self.isByzantine = False
        self.agentId = agentID 
        self.brain = neural_net
        self.value_function = value_function
        self.actionSpace = getHonestActionSpace(params)
        self.actionDims = len(self.actionSpace)
        self.stateDims = len(params['commit_vals'])+1 # +1 for the null value. 
        self.committed_ptr =  False
        self.reward = 0

        self.initVal = give_inits[agentID]
        # self.initState = self.initAgentState(params, init_val, give_inits)
        #self.state = torch.tensor(self.initState).float()
        self.state = self.initAgentState(params, init_val, give_inits)
        self.committed_value = False

        self.actionIndex = None
        self.actionString = ""

        # can use this to create agents that don't react to the policy
        self.action_callback = None
    
    def initAgentState(params, init_val, give_inits):
        initState = [init_val]
        for a in range(params['num_agents']-1):
            initState.append(params['null_message_val'])
        return torch.tensor(initState).uint8()

    def getHonestActionSpace(params):
        honest_action_space = getActionSpace(params, False, byzantine_inds=None, can_send_either_value=params['honest_can_send_either_value'])
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

    def update_agent_state(self, agent, agent_list):
        #Update the given agents state given what all the other agents actions are 
        actor_ind = 1
        new_state = [agent.initVal] # keep track of the agent's initial value
        for actor in agent_list:
            if agent.agentID == actor.agentID:
                continue
            new_state.append(actionEffect(self.params, actor.actionStr, actor.initVal, agent.state[actor_ind], actor.agentId))
            actor_ind +=1
        agent.state = torch.tensor(new_state).uint8()








