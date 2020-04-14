import copy
import numpy as np 
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import torch


class MultiAgentPPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    size: the number of rounds in the trajectory
    obs_dim: dimension of obs
    act_dim: actions
    """
    # self.honest_buffer = PPOBuffer(self.stateDims, 1, self.local_actions_per_epoch, params['num_agents']-params['num_byzantine'], gamma=params['gamma'], lam=params['lam'])

    def __init__(self, obs_dim, act_dim, size, num_agents, gamma=0.99, lam=0.95):
        self.obs_buf = [] #np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = [] #np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = [] #np.zeros(size, dtype=np.float32)
        self.rew_buf = [] #np.zeros(size, dtype=np.float32)
        self.ret_buf = [] #np.zeros(size, dtype=np.float32)
        self.val_buf = [] #np.zeros(size, dtype=np.float32)
        self.logp_buf = [] #np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.num_agents = num_agents

        # temp dict to compute everything for each agent. 
        store_dict = {'obs':[], 'act':[], 
        'rew':[], 'val':[], 'logp':[] }
        self.temp_buf = {i:copy.deepcopy(store_dict) for i in range(self.num_agents)}

    def store(self, agent_ind, obs, act, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # print("size: ",len(obs.numpy()))
        self.temp_buf[agent_ind]['obs'].append(obs.numpy())
        # print("obs in buf len: ", len(self.temp_buf[agent_ind]['obs']))
        self.temp_buf[agent_ind]['act'].append(act)
        self.temp_buf[agent_ind]['val'].append(val)
        self.temp_buf[agent_ind]['logp'].append(logp)

    def store_reward(self, agent_ind, rew):
        self.temp_buf[agent_ind]['rew'].append(rew)

    def finish_sim(self, agent_list):
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
        
        for ind, agent in enumerate(agent_list): # indices and dictionaries for each agent. 
            store_dic = self.temp_buf[ind]
            # print("agent reward: ", agent.reward)
            store_dic['obs'].append(agent.last_action_etc['obs'].numpy())
            store_dic['act'].append(agent.last_action_etc['act'])
            store_dic['val'].append(agent.last_action_etc['val'])
            store_dic['logp'].append(agent.last_action_etc['logp'])

            ##Add last reward
            store_dic['rew'].append(agent.reward) # adding the final reward that corresponds to each agents commit. has to be delayed until after each agent is finished. 

            store_dic['rew'].append(0)
            store_dic['val'].append(0)
            rews = np.asarray(store_dic['rew']) # adding the very last value. 
            vals = np.asarray(store_dic['val'])
            
            # the next two lines implement GAE-Lambda advantage calculation.
            # this is much more sophisticated than the basic advantage equation. 
            # print("rews: ", rews[:-1])
            # print("vals: ", vals[1:])
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            adv = core.discount_cumsum(deltas, self.gamma * self.lam)
            # print("rew length: ", len(rews))
            # print("vals length: ", len(vals))
            # print("adv length: ", adv.size)
            # print("obs length: ", len(store_dic['obs']))
            # print("act length: ", len(store_dic['act']))
            # print("logp length: ", len(store_dic['logp']))

            # the next line computes rewards-to-go, to be targets for the value function
            ret = core.discount_cumsum(rews, self.gamma)[:-1]
            # print(adv.size)
            self.obs_buf+= store_dic['obs']
            self.act_buf+= store_dic['act']
            self.rew_buf+= store_dic['rew'] # the actual reward recieved. 
            self.val_buf+= store_dic['val'] # the value function estimate. 
            self.logp_buf+= store_dic['logp']
            self.adv_buf += adv.tolist()
            self.ret_buf += ret.tolist()
            self.ptr += len(store_dic['obs']) # number of new observations added here. 

            #print('finish path data', self.obs_buf, self.ptr)

        store_dict = {'obs':[], 'act':[], 
        'rew':[], 'val':[], 'logp':[] }
        self.temp_buf = {i:copy.deepcopy(store_dict) for i in range(self.num_agents)}

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """

        #convert all to numpy array and then torch. 
        self.obs_buf = np.asarray(self.obs_buf)
        self.act_buf = np.asarray(self.act_buf) 
        self.rew_buf = np.asarray(self.rew_buf) # the actual reward recieved. 
        self.val_buf = np.asarray(self.val_buf) # the value function estimate. 
        self.logp_buf = np.asarray(self.logp_buf)
        self.adv_buf = np.asarray(self.adv_buf)

        #print('advantage buffer', self.adv_buf, type(self.adv_buf))

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

        #print('before the reset', data['obs'])
        
        # can now wipe the buffer? 
        self.obs_buf = [] #np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = [] #np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = [] #np.zeros(size, dtype=np.float32)
        self.rew_buf = [] #np.zeros(size, dtype=np.float32)
        self.ret_buf = [] #np.zeros(size, dtype=np.float32)
        self.val_buf = [] #np.zeros(size, dtype=np.float32)
        self.logp_buf = [] #np.zeros(size, dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0

        #print('after the reset', data['obs'])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

        '''# TODO: find a way to store this so dont have to do this every time it is called in the loss update!!
        # probably best to ultimately have all honest and byz in the same buffer... 
        assert self.ptr == self.max_size    # buffer has to be full before you can get. what if it ends early???
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        '''

    '''def reset(self):
        self.ptr, self.path_start_idx = 0, 0'''