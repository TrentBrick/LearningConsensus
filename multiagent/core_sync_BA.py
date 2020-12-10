import numpy as np 
from consensus_env import getActionSpace, actionEffect, sync_BA_effect  
import itertools
from collections import OrderedDict
from enum import Enum
import heapq


import torch
    


class Honest_Agent:

    def __init__(self, params, agentId, give_inits):
        self.isLeader = False
        self.isByzantine = False
        self.agentId = agentId
        self.stateDims = len(params['commit_vals'])+1 # +1 for the null value. 
        self.committed_ptr =  False
        self.reward = 0

        self.proposal = params['null_message_val']

        self.commitMessage = params['null_message_val']
        self.committedValue = params['null_message_val']

        self.accepted = Accepted()

        self.state = self.initAgentState(params, give_inits)
        self.roundValue = params['null_message_val']
        self.roundMessages = []
        self.statusValues = []
        self.messages = []

        self.actionIndex = None
        self.action = ""
        self.prevAction = ""

        self.last_action_etc = dict()
        # can use this to create agents that don't react to the policy
        self.action_callback = True
    
    def initAgentState(self, params, give_inits):
        initState = []
        for a in range(params['num_agents']):
            initState.append(params['null_message_val'])
        return torch.tensor(initState).int()


class Byzantine_Agent:

    def __init__(self, params, agentId, give_inits, byzantine_inds, is_leader):
        self.isByzantine = True
        self.agentId = agentId
        self.actionSpace = self.getByzantineActionSpace(params, byzantine_inds)
        self.actionDims = len(self.actionSpace)
        self.stateDims = len(params['commit_vals'])+1 # +1 for the null value. 
        self.committed_ptr =  False
        self.reward = 0
        self.isLeader = is_leader

        # Initialize action possibilities
        self.actionDict = dict.fromkeys(self.actionSpace)
        for key in self.actionDict:
            self.actionDict[key] = []
        self.actionDict['no_send'] = [Message(MessageType.NOSEND)]

        self.proposal = params['null_message_val'] 

        self.committedValue = params['null_message_val']
        self.commitMessage = params['null_message_val']

        self.accepted = Accepted()

        self.roundValue = params['null_message_val']
        self.roundMessages = []
        self.statusValues = []
        self.messages = []


        self.actionIndex = None
        self.actions = []
        self.prevAction = []
        self.actionString = ""
        self.prevActionString = ""

        self.last_action_etc = dict()
        # can use this to create agents that don't react to the policy
        self.action_callback = None
        state = self.initAgentState(params, give_inits)
        if self.isLeader:
            state.append(1)
        else:
            state.append(0)

        ### We don't want the agent to send anything in the FIRST status round, so zero out everything but no_send ###
        state.append(0)
        state = (state + [1]*70)

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
         # remove only yourself.
        non_byzantines = list(range(0, params['num_agents']))
        for agent_ind in non_byzantines:
            if agent_ind == self.agentId:
                non_byzantines.remove(agent_ind)
        # for byzantine_ind in byzantine_inds:
        #     if byzantine_ind in non_byzantines:
        #         non_byzantines.remove(byzantine_ind)

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

class Message(object):

    def __init__(self, MessageType, value=2, iteration=-1, certificate=-1, sender=-1, receiver='BROADCAST'):
        self.messageType = MessageType
        self.value = value
        self.iteration = iteration
        self.certificate = certificate
        self.sender = sender
        self.receiver = receiver
    
    def __str__(self):
        messageString = str(self.messageType) + ", " + "value-" + str(self.value) + ", " + str(self.iteration) + ", " + str(self.certificate) + ", " + str(self.sender) + ", " + str(self.receiver)
        return messageString
    
    def __lt__(self, other):
        return self.iteration < other.iteration

    def __cmp__(self, other):
        return self.iteration < other.iteration

    def __repr__(self):
        return str(self)

class MessageType(Enum):
    STATUS = 1
    PROPOSE = 2
    VOTE = 3
    COMMIT = 4
    NOSEND = 5
    NOCOMMIT = 6

class Accepted(object):

    def __init__(self, value=2, iteration=-1, certificate=-1):
        self.value = value
        self.iteration = iteration
        self.certificate = certificate
    
    def __str__(self):
        messageString = "value-" + str(self.value) + ", " + str(self.iteration) + ", " + str(self.certificate)
        return messageString
    
    def __lt__(self, other):
        return self.iteration < other.iteration

    def __cmp__(self, other):
        return self.iteration < other.iteration
    
    def __repr__(self):
        return str(self)
    
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

    def step(self, curr_sim_len, iteration):
        if (curr_sim_len%4 != 0):
            for agent in self.agents:
                self.update_agent_state(agent, self.agents, curr_sim_len, iteration)
        if curr_sim_len%4 == 0:
            for agent in self.honest_agents:
                new_state = [2] * self.params['num_agents']
                #Update vote counts for honest agents
                agent.state = torch.tensor(new_state).int()            

    def update_agent_state(self, agent, agent_list, curr_sim_len, iteration):
        #Update the given agents state given what all the other agents actions are 
        actor_ind = 1
        new_state = [agent.proposal.value] if agent.proposal != self.params['null_message_val'] else [self.params['null_message_val']]# keep track of the agent's initial value

        agent.roundMessages = [] # clear previous round messages

        old_state = agent.state.numpy().astype(int).tolist() # keep track of old state and turn into integer list
        old_state = old_state[0:self.params['num_agents']] # only want the first n elements - byznatine agent updates values later based on state

        ## Add received and send messsages from current round into byzantine agent's messages ## 
        if agent.isByzantine:
                for sending_agent in agent_list:
                    if not sending_agent.isByzantine:
                        if ('BROADCAST' == sending_agent.action.receiver or agent.agentId == sending_agent.action.receiver) and sending_agent.action.messageType != MessageType.NOSEND:
                            heapq.heappush(agent.messages, sending_agent.action)
                    else:
                        for act in agent.actions:
                            heapq.heappush(agent.messages, act)

        if curr_sim_len%4 == 1:
            #Only need to update the leader's state
            if agent.isLeader:
                agent.accepted = Accepted()
                # Update state
                for actor in agent_list:
                    if actor.agentId == agent.agentId:
                        if actor.isByzantine:
                            ##TODO: FIX this
                            agent.statusValues.append(2)
                            agent.roundMessages.append(Message(MessageType.NOSEND))
                        else:
                            agent.statusValues.append(actor.action.value)
                            agent.roundMessages.append(actor.action)
                        continue
                    if actor.isByzantine:
                        byzMessage = self.byzantine_round_message(agent, actor)
                        agent.roundMessages.append(byzMessage)
                        status = byzMessage.value
                    else:
                        status = actor.action.value
                        agent.roundMessages.append(actor.action)
                    new_state.append(status)
                    agent.statusValues.append(status)
                    actor_ind +=1

                #Update accepted value of leader
                quorum = (self.params['num_agents']+1)/2
                if len(agent.statusValues) >= quorum:
                    zeroCount = 0
                    oneCount = 0
                    for val in agent.statusValues:
                        if val == 0:
                            zeroCount += 1
                        if val == 1:
                            oneCount += 1
                    if zeroCount >= quorum or oneCount >= quorum:
                        proposeValue = 0 if zeroCount >= quorum else 1
                        # Check if we have f+1 messages with same value 
                        maximumMessage = Message(MessageType.NOSEND)
                        for message in agent.roundMessages:
                            if message.iteration > maximumMessage.iteration and message.value == proposeValue:
                                maximumMessage = message
                        agent.accepted = Accepted(maximumMessage.value, maximumMessage.iteration, maximumMessage.certificate)
            # Non leader's state remains the same
            else:
                new_state = old_state

        if curr_sim_len%4 == 2:
            # Leader's state remains the same because the leader is proposing - only update proposeValue
            equiv = False
            if agent.isLeader: 
                new_state = old_state
                new_state[0] = agent.proposal.value if agent.proposal != self.params['null_message_val'] else self.params['null_message_val']
            # Non leaders must update their propose value
            else:                
                for actor in agent_list:
                    # If the agent is itself, ignore
                    if agent.agentId == actor.agentId:
                        continue
                    # Append old state value
                    new_state.append(old_state[actor_ind])
                    # Update propose value to value sent from leader
                    if actor.isLeader:
                        if actor.isByzantine:
                            agent.proposal = self.byzantine_round_message(agent, actor)
                        else:
                            agent.proposal = actor.action
                        new_state[0] = agent.proposal.value #Update value in state to count votes in commit round
                    actor_ind +=1
        
        if curr_sim_len%4 == 3:
            #In vote round, everyone's state is updated
            for actor in agent_list:
                if agent.agentId == actor.agentId:
                    if agent.isByzantine:
                        for action in agent.actions:
                            agent.roundMessages.append(action)
                    else:
                        agent.roundMessages.append(actor.action)
                    continue
                if actor.isByzantine:
                    new_state.append(self.byzantine_action_result(agent, actor))
                    agent.roundMessages.append(self.byzantine_round_message(agent, actor))
                else:
                    new_state.append(actor.action.value)
                    agent.roundMessages.append(actor.action)
                actor_ind +=1

        #Update initial value - can't override this value later when the byzantine sends it 
        if curr_sim_len%4 == 2 and not agent.isLeader:
            for val in new_state:
                if val != self.params['null_message_val']:
                    new_state[0] = val
                    agent.proposeValue = val
        
        # if not agent.isByzantine and agent.isLeader:
        #     if curr_sim_len%4 == 1:
        #         if agent.state[1] == agent.state[2] and agent.state[1] != -1:
        #             agent.proposeValue = agent.state[1]
        #         else:
        #             agent.proposeValue = np.random.choice([0,1])
            
                    
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
            ## Check action possibilities 
            self.get_action_possibilities(agent, agent_list, iteration, curr_sim_len)
            for key, val in agent.actionDict.items():
                if len(val) == 0: #Action isn't possible
                    # new_state.append(1)
                    agent.actionDict[key].append(Message(MessageType.NOSEND))
                    new_state.append(0)
                else:
                    new_state.append(0)

            
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
            
            self.get_action_possibilities(agent, agent_list, iteration, curr_sim_len)
            for key, val in agent.actionDict.items():
                if len(val) == 0: #Action isn't possible
                    # new_state.append(1)
                    agent.actionDict[key].append(Message(MessageType.NOSEND))
                    new_state.append(0)
                else:
                    new_state.append(0)
        
        


        # print(new_state)
        agent.state = torch.tensor(new_state).int()

    def byzantine_action_result(self, agent, byzantineAgent):
        sentMessage = False
        for byzantineAction in byzantineAgent.actions:
            if 'BROADCAST' == byzantineAction.receiver or agent.agentId == byzantineAction.receiver and byzantineAction.messageType != MessageType.NOSEND:
                return byzantineAction.value
        return self.params['null_message_val']
    
    def byzantine_round_message(self, agent, byzantineAgent):
        sentMessage = False
        for byzantineAction in byzantineAgent.actions:
            if 'BROADCAST' == byzantineAction.receiver or agent.agentId == byzantineAction.receiver and byzantineAction.messageType != MessageType.NOSEND:
                return byzantineAction
        return Message(MessageType.NOSEND)

    def get_action_possibilities(self, agent, agent_list, iteration, curr_sim_len):
        agent.actionDict = dict.fromkeys(agent.actionSpace)
        for key in agent.actionDict:
            agent.actionDict[key] = []
        agent.actionDict['no_send'].append(Message(MessageType.NOSEND))
        if agent.isLeader:
            if curr_sim_len%4 == 1:
                ## Can basically propose any value here 
                if curr_sim_len == 1:
                    for action in agent.actionSpace:
                        if action == 'no_send':
                            pass
                        for receiving_agent in agent_list:
                            receiver_id = receiving_agent.agentId
                            if 'agent-' + str(receiver_id) in action:
                                value = int(action.split('agent-'+str(receiver_id)+'_v-')[-1][0])
                                agent.actionDict[action].append(Message(MessageType.PROPOSE, value, iteration, self.params['null_message_val'],
                                    agent.agentId, receiver_id))
                else:
                    ### Find if there are f+1 status votes for same value
                    status_values = []
                    for honest_agent in self.honest_agents:
                        status_values.append(honest_agent.action.value)
                    quorum = (self.params['num_agents']+1)/2
                    oneCount = 0
                    zeroCount = 0
                    for val in status_values:
                        if val == 1:
                            oneCount+=1
                        if val == 0:
                            zeroCount+=0
                    if oneCount >= quorum or zeroCount >= quorum:
                        certificate = self.honest_agents[0].action
                        ## Agent must propose value received f+1 status votes for
                        value = 1 if oneCount >= quorum else 0
                        #TODO: instead of looping through all actions, just manually create actionstring and find index 
                        for action in agent.actionSpace:
                            doAction = True
                            if action == 'no_send':
                                pass
                            for receiving_agent in agent_list:
                                if receiving_agent.agentId == agent.agentId:
                                    continue
                                if 'agent-'+str(receiving_agent.agentId) not in action or 'v-'+str(value-1) in action:
                                    doAction = False
                            if doAction:
                                for receiving_agent in agent_list:
                                    receiver_id = receiving_agent.agentId
                                    if 'agent-' + str(receiver_id) in action:
                                        agent.actionDict[action].append(Message(MessageType.PROPOSE, value, iteration, certificate,
                                            agent.agentId, receiver_id))
                            
                    else:
                        ## Agent can go and propose anything
                        for action in agent.actionSpace:
                            if action == 'no_send':
                                pass
                            for receiving_agent in agent_list:
                                receiver_id = receiving_agent.agentId
                                if 'agent-' + str(receiver_id) in action:
                                    value = int(action.split('agent-'+str(receiver_id)+'_v-')[-1][0])
                                    agent.actionDict[action].append(Message(MessageType.PROPOSE, value, iteration, self.params['null_message_val'],
                                        agent.agentId, receiver_id))

            if curr_sim_len%4 == 2:
                for action in agent.actionSpace:
                    if action == 'no_send':
                        pass
                    for receiving_agent in agent_list:
                        receiver_id = receiving_agent.agentId
                        if 'agent-' + str(receiver_id) in action:
                            value = int(action.split('agent-'+str(receiver_id)+'_v-')[-1][0])
                            for proposeMessage in agent.actions:
                                if proposeMessage.value == value:
                                    agent.actionDict[action].append(Message(MessageType.VOTE, value, proposeMessage.iteration, proposeMessage,
                                        agent.agentId, receiver_id))
        else:
            if curr_sim_len%4 == 1:
                ## Can't propose anything - not leader
                pass
            if curr_sim_len%4 == 2:
                ## Can only vote for what was sent by byzantine leader
                for action in agent.actionSpace:
                    if action == 'no_send':
                        pass
                    for receiving_agent in agent_list:
                        receiver_id = receiving_agent.agentId
                        if 'agent-' + str(receiver_id) in action:
                            value = int(action.split('agent-'+str(receiver_id)+'_v-')[-1][0])
                            if agent.proposal.value == value:
                                agent.actionDict[action].append(Message(MessageType.VOTE, value, agent.proposal.iteration, agent.proposal,
                                    agent.agentId, receiver_id))
            for action in agent.actionSpace:
                if action == 'no_send':
                    pass
                agent.actionDict[action].append(Message(MessageType.NOSEND))





