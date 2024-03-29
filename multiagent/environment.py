import gym
import itertools
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
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

        ##TODO: change if not a byzantine agent
        self.observation_space = spaces.Box(0, 3, (len(self.byzantine_agents[0].state),), dtype=np.uint8)
        self.leader = self.byzantine_agents[0]
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

        ## Set the leader & view change in round 5
        # if curr_sim_len == 1:
        #     self.byzantine_agents[0].isLeader = True
        # if curr_sim_len == 5:
        #     self.byzantine_agents[0].isLeader = False
        #     index = np.random.choice([0,1])
        #     self.leader = self.honest_agents[index]
        #     self.honest_agents[index].isLeader = True
        if curr_sim_len == 5:
            self.byzantine_agents[0].isLeader = False
            index = np.random.choice([0,1])
            self.leader = self.honest_agents[index]
            self.honest_agents[index].isLeader = True

        self.scripted_agents = self.world.scripted_agents
        # set action for each agent
        for ind, agent in enumerate(self.agents):
            self._set_action(action_n[ind], agent)
            if agent.committed_value != self.params['null_message_val'] and len(agent.last_action_etc.keys())==0: # agent has committed and it has only just committed!! ie it doesnt have any dictoinary values yet. 
                agent.last_action_etc['obs'] = agent.state
                agent.last_action_etc['act'] = action_n[ind]
                agent.last_action_etc['val'] = v_list[ind]
                agent.last_action_etc['logp'] = logp_list[ind] 
        
        for agent in self.scripted_agents:
           self._set_scripted_action(agent, curr_sim_len)
            
        #record if the leader has equivocated
        if curr_sim_len%4 == 2 and self.leader.isByzantine and 'v-0' in self.leader.actionString and 'v-1' in self.leader.actionString:
            # self.world.byzantineEquivocate = True
            pass

        # advance world state
        self.world.step(curr_sim_len)
        # record reward for each agent
        sim_done, reward_n, safety_violation = self._get_reward(curr_sim_len)
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
            for agent in self.allAgents:
                agent.status_values = []
                agent.proposeValue = self.params['null_message_val']
                agent.statusValue = self.params['null_message_val']

        return obs_n, reward_n, done_n, info_n, sim_done, safety_violation

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
    def _set_action(self, action_index, agent):
        agent.actionIndex = action_index
        agent.prevActionString = agent.actionString
        agent.actionString = agent.actionSpace[action_index]
        
        ###If commit in agents action space, then commit
        if 'commit' in agent.actionString:
            agent.committed_value = int(agent.actionString.split('_')[1])
    
    def _set_scripted_action(self, agent, curr_sim_len):
        if curr_sim_len%4 == 1:
            # Status round #
            if agent.isLeader:
                agent.actionString = 'pass'
            else:
                if agent.committed_value != self.params['null_message_val']:
                    agent.actionString = 'send_agent-'+str(self.leader.agentId)+ '_v-' + str(agent.committed_value)
                    agent.roundValue = agent.committed_value
                else:
                    agent.actionString = 'pass'
            ## Reset propose values values and saved state

        if curr_sim_len%4 == 2:
            # Propose Round #
            if agent.isLeader:
                if agent.statusValue == self.params['null_message_val']:
                    agent.proposeValue = np.random.choice([0,1])
                    agent.actionString = 'send_to-all_'+str(agent.proposeValue)
                else:
                    agent.proposeValue = agent.statusValue
                    agent.actionString = 'send_to-all_'+str(agent.proposeValue)
            else:
                agent.actionString = 'pass'

        if curr_sim_len%4 == 3:
            # Vote Round #
            if agent.proposeValue != self.params['null_message_val']:
                agent.actionString = 'send_to-all_' + str(agent.proposeValue)
                agent.roundValue = agent.proposeValue
            # Agent will not vote for anything becuase it hasn't been proposed a value
            else:
                agent.actionString = 'pass'

        if curr_sim_len%4 == 0:
            # Commit Round #
            if self.world.byzantineEquivocate:
                agent.actionString = 'no_commit'

                ##Reset byzantine equivocate for next rollout
                self.world.byzantineEquivocate = False
            else:
                #Get majority value
                zeroCount = 0
                oneCount = 0
                for val in agent.state:
                    if int(val) == 0:
                        zeroCount+=1
                    if int(val) == 1:
                        oneCount+=1
                # Update agent commit value
                quorum = (self.params['num_agents']+1)/2
                if zeroCount >= quorum:
                    agent.actionString = 'commit_0'
                    agent.committed_value = 0
                elif oneCount >= quorum:
                    agent.actionString = 'commit_1'
                    agent.committed_value = 1
                else:
                    agent.actionString = 'no_commit'  

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