import gym
import itertools
from gym import spaces
from gym.envs.registration import EnvSpec
from multiagent.core_sync_BA import Message, MessageType, Accepted
import numpy as np
import heapq
# from multiagent.multi_discrete import MultiDiscrete
from multiagent.multi_utils import MultiAgentActionSpace, MultiAgentObservationSpace

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    def __init__(self, params, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.params = params
        ## self.agents are agents that are controlled by neural networks
        self.agents = self.world.policy_agents
        self.honest_agents = self.world.honest_agents
        self.byzantine_agents = self.world.byzantine_agents
        ## self.allAgents is all the agents in the simulation
        self.allAgents = self.world.agents

        self.majorityValue = world.majorityValue
        # need to use all agents for n
        self.n = len(world.agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        ### NEED TO UPDATE THIS EVERY TIME DEPENDING ON WHICH AGENT IS DOING WHAT ###
        self.action_space = spaces.Discrete(self.byzantine_agents[0].actionDims)

        ##TODO: change if not a byzantine agent
        self.observation_space = spaces.Box(0, 3, (len(self.byzantine_agents[0].state),), dtype=np.uint8)
        
        self.iteration = 0
        

        for agent in self.allAgents:
            if agent.isLeader:
                self.leader = agent
                break
        # self.leader = self.byzantine_agents[0]
        print("I am: ", self.leader)

        # self.observation_space = []

        # self.action_space = []
        # for agent in self.agents:
        #     # May be useful to have this because agents could have different state sizes in the future
        #     obs_dim = len(observation_callback(agent, self.world))
        #     self.observation_space.append(spaces.Box(low=0, high=2, shape=(obs_dim,), dtype=np.uint8))
        #     self.action_space.append(spaces.Discrete(self.agents[0].actionDims))

    def step(self, action_n, v_list, logp_list, curr_sim_len):

        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        if (curr_sim_len%4 == 1):
            self.iteration += 1

        ######### View Changes #########
        ## View Change in round 5
        # if curr_sim_len == 5:
        #     self.byzantine_agents[0].isLeader = False
        #     index = np.random.choice([0,1])
        #     self.leader = self.honest_agents[index]
        #     self.honest_agents[index].isLeader = True
        if curr_sim_len == 5:
            # Remove current byzantine leader
            self.byzantine_agents[0].isLeader = False
            # Choose an honest leader
            index = np.random.choice([0,len(self.honest_agents)-1])
            self.leader = self.honest_agents[index]
            self.honest_agents[index].isLeader = True

        ##### View Changes Complete #####

        self.scripted_agents = self.world.scripted_agents
        # set action for each agent
        for ind, agent in enumerate(self.agents): # Agents controlled by neural network 
            self._set_action(action_n[ind], agent, curr_sim_len)
            if agent.committedValue != self.params['null_message_val'] and len(agent.last_action_etc.keys())==0: # agent has committed and it has only just committed!! ie it doesnt have any dictoinary values yet. 
                agent.last_action_etc['obs'] = agent.state
                agent.last_action_etc['act'] = action_n[ind]
                agent.last_action_etc['val'] = v_list[ind]
                agent.last_action_etc['logp'] = logp_list[ind] 
        
        for agent in self.scripted_agents: #Scripted agent
           self._set_scripted_action(agent, curr_sim_len)

        # Record if the leader has equivocated
        if (curr_sim_len%4 == 2 or curr_sim_len%4 == 3) and self.leader.isByzantine and 'v-0' in self.leader.actionString and 'v-1' in self.leader.actionString:
            # self.world.byzantineEquivocate = True
            pass

        # Check if Byzantine Agent did not propose correct value
        if self.leader.isByzantine and curr_sim_len == 6: 
            propose_values = []
            for agent in self.scripted_agents:
                propose_values.append(agent.roundValue)
            correct_propose_value = False
            quorum = (self.params['num_agents']+1)/2
            #Find Count
            zeroCount = 0
            oneCount = 1
            for propose_val in propose_values:
                if propose_val == 0:
                    zeroCount+=1
                if propose_val == 1:
                    oneCount+=1
            if zeroCount >= quorum or oneCount >= quorum:
                correct_propose_value = 0 if zeroCount >= quorum else 1
                incorrect_propose_value = 1 - correct_propose_value
                if 'v-'+str(incorrect_propose_value) in self.leader.actionString:
                    self.world.byzantineIncorrectPropose = True

        # advance world state
        self.world.step(curr_sim_len, self.iteration)
        # record reward for each agent
        sim_done, reward_n, safety_violation, delay_termination, safety_termination = self._get_reward(curr_sim_len)
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))


        # info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case - may be good to add in the future
        # reward = np.sum(reward_n)
        # if self.shared_reward:
        #     reward_n = [reward] * self.n

        #Reset view specific state
        if curr_sim_len%4 == 0:
            # Reset byzantine equivocate/incorrect propose for next round
            self.world.byzantineEquivocate = False
            self.world.byzantineIncorrectPropose = False
            for agent in self.allAgents:
                agent.statusValues = []
                agent.proposal = self.params['null_message_val']

        return obs_n, reward_n, done_n, info_n, sim_done, safety_violation, delay_termination, safety_termination

    def reset(self):
        # reset world
        self.reset_callback(self.params, self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        self.honest_agents = self.world.honest_agents
        self.byzantine_agents = self.world.byzantine_agents
        self.allAgents = self.world.agents

        self.majorityValue = self.world.majorityValue

        self.iteration = 0

        #Reset leader
        for agent in self.allAgents:
            if agent.isLeader:
                self.leader = agent
                break

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n    

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # we are returning whether each agent is done and also sim_done for get_reward
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent)

    # get rewards for all agents
    def _get_reward(self, curr_sim_len):
        return self.reward_callback(self.params, curr_sim_len, self.world)

    # set env action for a particular agent - this still needs to be configured
    def _set_action(self, action_index, agent, curr_sim_len):
        if len(agent.actionDict['no_send']) > 1:
            del agent.actionDict['no_send'][-1] 
        agent.prevAction = agent.actions
        agent.prevActionString = agent.actionString
        agent.actionIndex = action_index
        agent.actionString = agent.actionSpace[action_index]
        agent.actions = agent.actionDict[agent.actionString]
        # if curr_sim_len == 0:
        #         print("actionIndex: ", action_index)
        #         print("actionString: ", agent.actionString)
        #         print("actions: ", agent.actions)
    
    def _set_scripted_action(self, agent, curr_sim_len):
        if curr_sim_len%4 == 1:
            # Notify
            if agent.committedValue == self.params['null_message_val']:
                for committed_agent in self.scripted_agents:
                    if committed_agent.agentId == agent.agentId:
                        pass
                    if committed_agent.committedValue != self.params['null_message_val']:
                        agent.accepted = committed_agent.accepted

            # Status round #
            ### New Code ###
            if agent.accepted.value != self.params['null_message_val']:
                agent.action = Message(MessageType.STATUS, agent.accepted.value, agent.accepted.iteration,
                    agent.accepted.certificate, agent.agentId, self.leader.agentId)
                agent.roundValue = agent.accepted.value
            else:
                agent.action = Message(MessageType.NOSEND)
                agent.roundValue = self.params['null_message_val']

        if curr_sim_len%4 == 2:
            # Propose Round #
            #### New Code ####
            if agent.isLeader:
                # Choose accepted value
                if agent.accepted.value != self.params['null_message_val']:
                    proposeValue = agent.accepted.value
                    certificate = agent.accepted.certificate
                    iteration = agent.accepted.iteration
                # Choose proposal freely
                else:
                    proposeValue = np.random.choice([0,1])
                    certificate = self.params['null_message_val']
                    iteration = self.iteration 
                # Broadcast proposal
                agent.action = Message(MessageType.PROPOSE, proposeValue, iteration,
                    certificate, agent.agentId, "BROADCAST")
                agent.proposal = agent.action
            else:
                agent.action = Message(MessageType.NOSEND)

        if curr_sim_len%4 == 3:
            # Vote Round #
            #### New Code ####
            if agent.proposal.messageType != MessageType.NOSEND:
                agent.action = Message(MessageType.VOTE, agent.proposal.value, agent.proposal.iteration,
                    agent.proposal.certificate, agent.agentId, "BROADCAST")
            else:
                agent.action = Message(MessageType.NOSEND)

        if curr_sim_len%4 == 0:
            # Commit Round #
            agent.accepted = Accepted() # Reset accepted value

            if agent.committedValue != self.params['null_message_val']:
                # agent.commitMessage.iteration = iteration
                agent.action = agent.commitMessage
                agent.accepted = Accepted(agent.commitMessage.value, agent.commitMessage.iteration, agent.commitMessage.certificate)

            # elif self.world.byzantineEquivocate or self.world.byzantineIncorrectPropose: 
            #     agent.action = Message(MessageType.NOCOMMIT)

            elif not self.world.byzantineEquivocate and not self.world.byzantineIncorrectPropose:
                zeroCount = 0
                oneCount = 0
                for message in agent.roundMessages:
                    if message.value == 0:
                        zeroCount+=1
                    if message.value == 1:
                        oneCount+=1
                quorum = (self.params['num_agents']+1)/2
                if zeroCount >= quorum or oneCount >= quorum:
                    agent.committedValue = 0 if zeroCount >= quorum else 1
                    agent.roundValue = agent.committedValue
                    ## Create certificate
                    certificate = []
                    for message in agent.roundMessages:
                        if message.value == agent.committedValue:
                            certificate.append(message)
                    
                    agent.action = Message(MessageType.COMMIT, agent.committedValue, self.iteration, 
                        certificate, agent.agentId, "BROADCAST")                    
                    agent.commitMessage = agent.action
                    agent.accepted = Accepted(agent.commitMessage.value, agent.commitMessage.iteration, agent.commitMessage.certificate)

                else:
                    agent.action = Message(MessageType.NOCOMMIT)
                    agent.roundValue = self.params['null_message_val']                    
        


    # reset rendering assets
    # def _reset_render(self):
    #     self.render_geoms = None
    #     self.render_geoms_xform = None

    # render environment
    # def render(self, mode='human'):
    #     return results


# vectorized wrapper for a batch of multi-agent environments - can maybe use this for just honest agents
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):

    # TODO: should these be here? what does human render mean? 
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    # def render(self, mode='human', close=True):
    #     results_n = []
    #     for env in self.env_batch:
    #         results_n += env.render(mode, close)
    #     return results_n