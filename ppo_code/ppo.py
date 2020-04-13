import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
import gym
import time
from consensus_env import onehotter
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

def ppo_algo(env, seed=0, 
        actions_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=5):
    """f
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        actions_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    honest_logger = EpochLogger(**logger_kwargs)
    #honest_logger.save_config(locals())
    '''byzantine_logger = EpochLogger(**logger_kwargs)
    byzantine_logger.save_config(locals())'''
    #logger = EpochLogger(**logger_kwargs)
    #logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    #env = env_fn()
    ### Observation dimensions 
    #obs_dim = env.oneHotStateMapper.shape()
    # obs_dim = env.observation_space.shape
    #honest_act_dim = env.honest_oneHotActionMapper.shape()
    #byz_action_dim = env.byzantine_oneHotActionMapper.shape()
    # act_dim = env.action_space.shape

    # Create actor-critic module. # brains of each of the agents.
    #ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(env.honest_policy)
    sync_params(env.byz_policy)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [env.honest_policy, env.byz_policy, 
                                                            env.honest_v_function, env.byz_v_function])
    # honest_logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer - This is done in consensus_env.py
    #simulations_per_epoch = int(actions_per_epoch / num_procs()) # how do local steps fill up the buffer??
    #buf = PPOBuffer(env.stateDims, 1, simulations_per_epoch, gamma=params['gamma'], lam=params['lam'])
    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, nn, stateDims):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        # TODO: need to be able to input all of the observations and compute their logp and the action. 
        
        # get prob dist for the observation: 
        #print('observation for compute loss pi', obs, obs.shape)
        oh = onehotter(obs, stateDims)
        #print(oh.shape)
        logits = nn(oh)
        prob_dist = torch.nn.functional.softmax(logits, dim=1)
        log_prob_dist = torch.log(prob_dist)
        logp = torch.gather(log_prob_dist, 1, act.unsqueeze(1).long()) 
        #logp = torch.log(prob_dist[act])

        ratio = torch.exp(logp - logp_old)
        #print( 'ratio', ratio, ratio.shape, 'advantage',  adv, adv.shape)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()

        # homemade entropy calc
        # need to allow for numerical stability. nans propagate. 
        unq_ents = prob_dist * (log_prob_dist+0.000000000001) # elementwise multiplication of the probabilities
        ent = - unq_ents.mean().item() 
        #ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data, nn, stateDims):
        obs, ret = data['obs'], data['ret']
        oh = onehotter(obs, stateDims)
        #print('obs', obs, obs.shape, 'ret', ret, ret.shape)
        #print('nn oh v output', nn(oh), nn(oh).shape)
        return ((nn(oh) - ret)**2).mean()

    # Set up model saving
    # TODO: find a way to log both of the neural networks. 
    #byzantine_logger.setup_pytorch_saver(env.byz_policy)
    honest_logger.setup_pytorch_saver(env.honest_policy)

    def update(buf, nn, vf, pi_optimizer, vf_optimizer, stateDims):
        data = buf.get()

        #print('data that is collected before the update', data)

        pi_l_old, pi_info_old = compute_loss_pi(data, nn, stateDims)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data, vf, stateDims).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data, nn, stateDims)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                honest_logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(nn)    # average grads across MPI processes
            pi_optimizer.step()

        honest_logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data, vf, stateDims)
            loss_v.backward()
            mpi_avg_grads(vf)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        honest_logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    # TODO: see if I need to keep any of these return functions. 
    o, ep_ret, ep_len = env.reset(), 0, 0 

    local_actions_per_epoch = env.local_actions_per_epoch
    
    print(' local actions per epoch', local_actions_per_epoch)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        sim_done = False
        curr_ep_trajectory_logs = []
        single_run_trajectory_log = setup_trajectory_log(env)
        while env.majority_agent_buffer.ptr < local_actions_per_epoch and not sim_done:     
            '''' not sure if I want the neural networks here in ppo. 
            no I want the updates to happen within the agents themselves. '''
            sim_done, v, trajectory_log = env.env_step(single_run_trajectory_log)#, honest_logger, byzantine_logger) # episode length and then the total number of steps in the buffer. 
            single_run_trajectory_log = trajectory_log
            honest_logger.store(VVals=v)
            if sim_done:
                #for a in env.honest_list: 
                    #print(a.actionStr, a.committed_value)
                #print('========')
                curr_ep_trajectory_logs.append(single_run_trajectory_log)
                single_run_trajectory_log = setup_trajectory_log(env)
                o, ep_ret, ep_len = env.resetStatesandAgents(), 0, 0 # reset the environment

            #print('finished simulation', t)

        if epoch+1 % 5==0:
            print('======= end of simulations for epoch:', epoch+1)

        # TODO: get model save working. 
        # Save model
        #if (epoch % save_freq == 0) or (epoch == epochs-1):
            #logger.save_state({'env': env}, None)

        # Perform PPO update!
        if env.params['num_agents'] - env.params['num_byzantine'] > 0: 
            update(env.honest_buffer, env.honest_policy, env.honest_v_function,
            env.honest_optimizer, env.honest_v_function_optimizer, env.stateDims)
        if env.params['num_byzantine']>0:
            update(env.byz_buffer, env.byz_policy, env.byz_v_function, 
            env.byz_optimizer, env.byz_v_function_optimizer, env.stateDims)
        

        # Log info about epoch
        # TODO: get the logger to work
        
        '''if updating_byzantine_network:
            byzantine_logger.log_tabular('Epoch', epoch)
            byzantine_logger.log_tabular('EpRet', with_min_and_max=True)
            byzantine_logger.log_tabular('EpLen', average_only=True)
            byzantine_logger.log_tabular('VVals', with_min_and_max=True)
            byzantine_logger.log_tabular('TotalEnvInteracts', (epoch+1)*actions_per_epoch)
            byzantine_logger.log_tabular('LossPi', average_only=True)
            byzantine_logger.log_tabular('LossV', average_only=True)
            byzantine_logger.log_tabular('DeltaLossPi', average_only=True)
            byzantine_logger.log_tabular('DeltaLossV', average_only=True)
            byzantine_logger.log_tabular('Entropy', average_only=True)
            byzantine_logger.log_tabular('KL', average_only=True)
            byzantine_logger.log_tabular('ClipFrac', average_only=True)
            byzantine_logger.log_tabular('StopIter', average_only=True)
            byzantine_logger.log_tabular('Time', time.time()-start_time)
            byzantine_logger.dump_tabular()
        else:'''
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            print('=============================')
            print('last trajectory from this epoch:')
            for k, v in curr_ep_trajectory_logs[-1].items():
                print(k, v)
                print('---------')
            print('=============================')

        honest_logger.log_tabular('Epoch', epoch)
        #honest_logger.log_tabular('EpRet', with_min_and_max=True)
        #honest_logger.log_tabular('EpLen', average_only=True)
        honest_logger.log_tabular('VVals', with_min_and_max=True)
        honest_logger.log_tabular('TotalEnvInteracts', (epoch+1)*actions_per_epoch)
        honest_logger.log_tabular('LossPi', average_only=True)
        honest_logger.log_tabular('LossV', average_only=True)
        honest_logger.log_tabular('DeltaLossPi', average_only=True)
        honest_logger.log_tabular('DeltaLossV', average_only=True)
        honest_logger.log_tabular('Entropy', average_only=True)
        honest_logger.log_tabular('KL', average_only=True)
        honest_logger.log_tabular('ClipFrac', average_only=True)
        honest_logger.log_tabular('StopIter', average_only=True)
        honest_logger.log_tabular('Time', time.time()-start_time)
        honest_logger.dump_tabular()

        # need to reset the agents and whole environment between epochs. 
        o, ep_ret, ep_len = env.reset(), 0, 0

'''if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, actions_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)'''
def setup_trajectory_log(env):
    single_run_trajectory_log = dict()
    for agent in env.agent_list:
        single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentID)] = []
    return single_run_trajectory_log