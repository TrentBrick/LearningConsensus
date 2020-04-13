import gym
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
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
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

        # configure action space and observation for each agent - using MultiAgentUtils
        # TODO: does this work for Byzantine agents? they have a different sized action space. 
        # TODO: why is this a list with all the vectors for all the agents? 
        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.agents[0].actionDims) for _ in range(self.n)])
        # each agent has a vector of ints for the actions of themself and all other agents. it is in an int but call Box not discrete?? 
        self.observation_space = MultiAgentObservationSpace([spaces.Box(0, 2, (self.n,), dtype=np.uint8) for _ in range(self.n)])
        
        # self.observation_space = []
        # self.action_space = []
        # for agent in self.agents:
        #     # May be useful to have this because agents could have different state sizes in the future
        #     obs_dim = len(observation_callback(agent, self.world))
        #     self.observation_space.append(spaces.Box(low=0, high=2, shape=(obs_dim,), dtype=np.uint8))
        #     self.action_space.append(spaces.Discrete(self.agents[0].actionDims))

        ###############################
        # for agent in self.agents:
        #     total_action_space = []
        #     # physical action space
        #     if self.discrete_action_space:
        #         u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
        #     else:
        #         u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
        #     if agent.movable:
        #         total_action_space.append(u_action_space)
        #     # communication action space
        #     if self.discrete_action_space:
        #         c_action_space = spaces.Discrete(world.dim_c)
        #     else:
        #         c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
        #     if not agent.silent:
        #         total_action_space.append(c_action_space)
        #     # total action space
        #     if len(total_action_space) > 1:
        #         # all action spaces are discrete, so simplify to MultiDiscrete action space
        #         if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
        #             act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
        #         else:
        #             act_space = spaces.Tuple(total_action_space)
        #         self.action_space.append(act_space)
        #     else:
        #         self.action_space.append(total_action_space[0])
        #     # observation space
        #     obs_dim = len(observation_callback(agent, self.world))
        #     self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
        #     agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        # self.shared_viewer = shared_viewer
        # if self.shared_viewer:
        #     self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n
        # self._reset_render()
        ###################################

    def step(self, action_n, curr_sim_len):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
        # advance world state
        self.world.step()
        # record reward for each agent
        sim_done, reward_n = self._get_reward(curr_sim_len)
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))

        # info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case - may be good to add in the future
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n, sim_done

    def reset(self):
        # reset world
        self.reset_callback(self.params, self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
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
        agent.actionString = agent.actionSpace[action_index]
        
        ###If commit in agents action space, then commit
        if 'commit' in agent.actionString:
            agent.committed_value = int(agent.actionString.split('_')[1])

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