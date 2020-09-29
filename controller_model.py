import math
import random 
import time
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from policy import Policy
import pickle
#from ha_env import make_env
from multiagent.make_env_old import make_env as make_game_env
from torch.distributions import Categorical
#import gym
#import gym.envs.box2d

def flatten_parameters(params):
    """ Flattening parameters.
    :args params: generator of parameters (as returned by module.parameters())
    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device) # why werent these put on the device earlier? 
    idx = 0
    unflattened = []
    for e_p in example:
        # makes a list of parameters in the same format and shape as the network. 
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, policy):
    """ Load flattened parameters into policy.

    :args params: parameters as a single 1D np array
    :args policy: module in which params is loaded
    """
    proto = next(policy.parameters())
    params = unflatten_parameters(
        params, policy.parameters(), proto.device) # dont see the need to pass the device here only to put them into it later. 

    for p, p_0 in zip(policy.parameters(), params):
        p.data.copy_(p_0)

    return policy

class Models:

    def __init__(self, 
        mdir=None, return_events=False, give_models=None, parameters=None):
        """ policy and environment. """

        self.params = parameters
        print("parameters being used", parameters)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_limit = parameters['max_round_len']
        self.temperature = parameters['temperature']

        #print('loadin in policy.')
        self.make_env()
        print('env is:', self.env)
        self.policy = Policy(self.env.observation_space, self.env.action_space)
        print('######################################')
        print("action space is: ", self.env.agents[0].actionSpace)
        print('######################################')


        # load policy if it was previously saved
        '''
        ctrl_file = join(mdir, m, 'ctrl_best.tar')
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(self.device)})
            print("Loading policy with reward {}".format(
                ctrl_state['reward']))
            self.policy.load_state_dict(ctrl_state['state_dict'])'''

    def make_env(self):
        if self.params['scenario'] == 'honest_basic':
            self.env = make_game_env(self.params, "honest_basic")
                

        # need to alter the way that rewards are computed first. 
        '''elif params['scenario'] == 'honest_byzantine':
            env = make_env(self.params, "honest_byzantine")
            
        elif params['scenario'] == 'sync_BA':
            env = make_env(self.params, "sync_BA")'''

    def rollout(self, rand_env_seed, params=None, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the policy and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        # Why is this the minus cumulative reward?!?!!?
        """

        # copy params into the policy
        if params is not None:
            self.policy = load_parameters(params, self.policy)

        #random(rand_env_seed)
        np.random.seed(rand_env_seed)
        torch.manual_seed(rand_env_seed)
        self.make_env()
        o_list, honest_ep_ret, byzantine_ep_ret, ep_len = self.env.reset(), 0, 0, 0
        #self.env.seed(int(rand_env_seed)) # ensuring that each rollout has a differnet random seed. 
        #obs = self.env.reset()

        # This first render is required !
        #self.env.render()
        cumulative = 0
        round_len = 0
        #if self.return_events: 
        #    rollout_dict = {k:[] for k in ['obs', 'rew', 'act', 'term']}
        while True:

            #print('iteration of the rollout', round_len, rand_env_seed)

            # observations for each agent!!! Need to loop through... 

            actions = []
            for a_ind, agent in enumerate(self.env.agents):
                if agent.committed_value is False:
                    action_logits = self.policy(torch.as_tensor(o_list[a_ind], dtype=torch.float32))
                    action = Categorical(torch.softmax(action_logits, dim=1)/self.temperature).sample()
                    
                else: 
                    action = agent.actionIndex
                
                actions.append(action)

            for ind, agent in enumerate(self.env.agents):
                #print('actions decided are:', actions)
                agentActionString = agent.actionSpace[actions[ind]]
                if 'commit' in agentActionString:
                    #print('committed! agent action string:', agentActionString)
                    agent.committed_value = int(agentActionString.split('_')[1])

                #print('action string is:', agentActionString, ind, rand_env_seed)


            # CNA THEY NOT ALL STEP TOGETHER? 
            next_o, r_list, d_list, info_n_list, sim_done = self.env.step(actions, round_len)

            reward = sum(r_list)

            # Update obs (critical!)
            o_list = next_o

            '''if self.return_events: 
                for key, var in zip(['obs', 'rew', 'act', 'term'], [obs,reward, action, done]):
                    rollout_dict[key].append(var)'''

            cumulative += reward
            if sim_done or round_len > self.time_limit:
                #print('done with this simulation')
                '''if self.return_events:
                    for k,v in rollout_dict.items():
                        rollout_dict[k] = np.array(v)
                    return cumulative, rollout_dict
                else: '''
                return cumulative, round_len # ending time and cum reward
            round_len += 1

    def simulate(self, params, train_mode=True, 
        render_mode=False, num_episode=16, 
        seed=27): # run lots of rollouts 

        #print('seed recieved for this set of simulations', seed)
        # update params into the policy
        self.policy = load_parameters(params, self.policy)

        recording_mode = False
        penalize_turning = False

        #random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        reward_list = []
        t_list = []

        with torch.no_grad():
            for i in range(num_episode):

                rand_env_seed = np.random.randint(0,1e9,1)[0]
                #print('starting new rollout')
                rew, t = self.rollout(rand_env_seed, render=render_mode, 
                            params=None)
                #print('finished a rollout')
                reward_list.append(rew)
                t_list.append(t)

        return reward_list, t_list