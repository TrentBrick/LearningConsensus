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

        self.proposeValue = params['null_message_val']
        self.statusValue = params['null_message_val']
        self.committed_value = params['null_message_val']
        self.roundValue = params['null_message_val']
        self.status_values = []

        self.state = self.initAgentState(params, give_inits)

        self.actionIndex = None
        self.actionString = ""

        self.last_action_etc = dict()
        # can use this to create agents that don't react to the policy
        self.action_callback = True
    
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
        self.status_values = []
        self.committed_value = params['null_message_val']

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

        ### We don't want the agent to send anything in the FIRST status round, so zero out everything but no_send ###
        # state.append(0)
        # state = (state + [1]*8)

        # # 2 actions where send something different
        # state = (state + [0]*5)
        # state.append(1)
        # state.append(0)
        # state.append(1)
        # state.append(0)
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

        return action_space
        
    
#multi-agent world
class World(object):
    
    def __init__(self, params):
        self.params = params
    
    #return all agents controlled by a policy 
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self, curr_sim_len):
        if (curr_sim_len%4 != 0):
            for agent in self.agents:
                self.update_agent_state(agent, self.agents, curr_sim_len)
        if curr_sim_len%4 == 0:
            for agent in self.honest_agents:
                new_state = [2] * self.params['num_agents']
                #Update vote counts for honest agents
                agent.state = torch.tensor(new_state).int()            

    def update_agent_state(self, agent, agent_list, curr_sim_len):
        #Update the given agents state given what all the other agents actions are 
        actor_ind = 1
        new_state = [agent.proposeValue] # keep track of the agent's initial value

        old_state = agent.state.numpy().astype(int).tolist() # keep track of old state and turn into integer list
        old_state = old_state[0:self.params['num_agents']] # only want the first n elements - byznatine agent updates values later based on state

        if curr_sim_len%4 == 1:
            #Only need to update the leader's state
            if agent.isLeader:
                new_state = old_state
                if agent.committed_value != self.params['null_message_val']:
                    agent.status_values.append(agent.committed_value)
                for actor in agent_list:
                    if agent.agentId == actor.agentId:
                        continue
                    status = sync_BA_effect(self.params, self.agents, actor.actionString, agent.state[actor_ind], agent.agentId, curr_sim_len)
                    if status != 2:
                        agent.status_values.append(status)
                    actor_ind +=1
                #Update status value of agent
                if len(agent.status_values) == 0 or len(set(agent.status_values)) > 1:
                    agent.statusValue = self.params['null_message_val']
                else:
                    agent.statusValue = agent.status_values[0]

            # Non leader's state remains the same
            else:
                new_state = old_state

        if curr_sim_len%4 == 2:
            # Propose value for leader already updated in the set_scriped agent action in environment.py 

            # Leader's state remains the same because the leader is proposing - only update proposeValue
            if agent.isLeader: 
                new_state = old_state
                new_state[0] = agent.proposeValue
            # Non leaders must update their propose value
            else:
                for actor in agent_list:
                    # If the agent is itself, don't update
                    if agent.agentId == actor.agentId:
                        continue
                    # Append old state value
                    new_state.append(old_state[actor_ind])
                    # Update propose value to value sent from leader
                    if actor.isLeader:
                        agent.proposeValue = sync_BA_effect(self.params, self.agents, actor.actionString, agent.state[actor_ind], agent.agentId, curr_sim_len)
                        new_state[0] = agent.proposeValue #Update value in state to count votes in commit round
                    actor_ind +=1
        
        if curr_sim_len%4 == 3:
            #In vote round, everyone's state is updated
            for actor in agent_list:
                if agent.agentId == actor.agentId:
                    continue
                new_state.append(sync_BA_effect(self.params, self.agents, actor.actionString, agent.state[actor_ind], agent.agentId, curr_sim_len))
                actor_ind +=1

        #Update initial value - can't override this value later when the byzantine sends it 
        if curr_sim_len%4 == 2 and not agent.isLeader:
            for val in new_state:
                if val != self.params['null_message_val']:
                    new_state[0] = val
                    agent.proposeValue = val
        
        if not agent.isByzantine and agent.isLeader:
            if curr_sim_len%4 == 1:
                if agent.state[1] == agent.state[2] and agent.state[1] != -1:
                    agent.proposeValue = agent.state[1]
                else:
                    agent.proposeValue = np.random.choice([0,1])
            
                    
        if agent.isByzantine and agent.isLeader:
            oneCount = 0
            zeroCount = 0
            for actor in agent_list:
                if not agent.isByzantine:
                    if 'no_send' not in actor.actionString and 'commit' not in actor.actionString:
                        if agent.roundValue == 1:
                            oneCount += 1
                        if agent.roundValue == 0:
                            zeroCount +=1 
            quorum = (self.params['num_agents']+1)/2
            quorumVal = False
            if oneCount >= quorum:
                quorumVal = oneCount
            if zeroCount >= quorum:
                quorumVal = zeroCount

            #Append sim len
            if curr_sim_len == 4 or curr_sim_len == 8:
                new_state.append(4)
            else:
                new_state.append(curr_sim_len%4)
            #Append if leader or not
            if agent.isLeader:
                new_state.append(1)
            else:
                new_state.append(0)

            # ## These next methods are zero-ing out action probabilities for the next round ## 
            # if curr_sim_len%4 == 1:
            #     # If we get f+1 status votes for a single value, then we must propose that value in next round
            #     if quorumVal is not False:
            #         #Cancel out first 5 actions - no_send and sending individual agents
            #         new_state = (new_state + [1]*5)
            #         for action in agent.actionSpace[5:]:
            #             if 'v-0' in action and 'v-1' in action:
            #                 new_state.append(1)
            #             elif 'value-'+str(quorumVal) in action:
            #                 new_state.append(0)
            #             else:
            #                 new_state.append(1)
            #     else:
            #         #Balance for equivocation
            #         new_state = (new_state + [0]*5)
            #         new_state.append(1)
            #         new_state.append(0)
            #         new_state.append(1)
            #         new_state.append(0)
                    
            # if curr_sim_len%4 == 2:
            #     #Balance for equivocation
            #     new_state = (new_state + [0]*5)
            #     new_state.append(1)
            #     new_state.append(0)
            #     new_state.append(1)
            #     new_state.append(0)

            # if curr_sim_len%4 == 3:
            #     #Balance for equivocation
            #     new_state = (new_state + [0]*5)
            #     new_state.append(1)
            #     new_state.append(0)
            #     new_state.append(1)
            #     new_state.append(0)
            
        if agent.isByzantine and not agent.isLeader:
            # Find index of leader
            leaderId = -1
            for actor in agent_list:
                if actor.isLeader:
                    leaderId = actor.agentId
                    break
            
            #Append sim len
            if curr_sim_len%4 == 0:
                new_state.append(4)
            else:
                new_state.append(curr_sim_len%4)
            #Append if leader or not
            if agent.isLeader:
                new_state.append(1)
            else:
                new_state.append(0)

            # ## The next methods zero out actions that the agent should not do
            # if curr_sim_len%4 == 1:
            #     # Shouldn't be sending anything except no_send in the propose round
            #     new_state.append(0)
            #     new_state = (new_state + [1]*8)
            # if curr_sim_len%4 == 2:
            #     # Agent can only vote for what was proposed to it
            #     proposeStringOpposite = 'v-' + str(1 - agent.proposeValue)
            #     new_state.append(0)
            #     new_state = (new_state + [1]*4)
            #     for action in agent.actionSpace[5:]:
            #         if proposeStringOpposite in action:
            #             new_state.append(1)
            #         else:
            #             new_state.append(0)
            # if curr_sim_len%4 == 3:
            #     #Balance for equivocation
            #     new_state = (new_state + [0]*5)
            #     new_state.append(1)
            #     new_state.append(0)
            #     new_state.append(1)
            #     new_state.append(0)
            





        # print(new_state)
        agent.state = torch.tensor(new_state).int()








