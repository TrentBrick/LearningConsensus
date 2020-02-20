import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
# TODO: Import one hot state mapper

def ppo_algo(env, seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
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

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
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
    # byzantine_logger = EpochLogger(**logger_kwargs)
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
    #logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs()) # how do local steps fill up the buffer??
    #buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, nn):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = nn.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data, nn):
        obs, ret = data['obs'], data['ret']
        return ((nn.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    # pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    # vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    honest_policy_optimizer = env.honest_optimizer
    byzantine_policy_optimizer = env.byz_optimizer
    honest_v_optimizer = env.honest_v_function_optimizer
    byz_v_optimizer = env.byz_v_function_optimizer

    # Set up model saving
    # TODO: find a way to log both of the neural networks. 
    #logger.setup_pytorch_saver(env.honest_policy)
    ###Idea - create two logger classes - let's just focus on building this out for honest now ###
    honest_logger.setup_pytorch_saver(env.honest_policy)


    def update():

        for agent_type_list in [env.honest_list, env.byzantine_list]:

            pi_l_old_avg = 0
            pi_info_old_avg = dict()
            v_l_old_avg = 0

            for agent in agent_type_list: 

                data = agent.buffer.get() # get a dictionary format of the logger with the values normalized. 

                pi_l_old, pi_info_old = compute_loss_pi(data, agent.brain)
                pi_l_old = pi_l_old.item()
                v_l_old = compute_loss_v(data, agent.brain).item()

                # add to the averages
                v_l_old_avg += v_l_old
                pi_l_old_avg += pi_l_old
                if not pi_info_old_avg:
                    pi_info_old_avg = pi_info_old.copy()
                else: 
                    for k in pi_info_old.keys():
                        pi_info_old_avg[k] += pi_info_old[k]
                    
            # divide by number of agents: 
            pi_l_old_avg /= len(agent_type_list)
            v_l_old_avg /= len(agent_type_list)
            for k in pi_info_old_avg.keys():
                pi_info_old_avg[k] = pi_info_old_avg[k]/len(agent_type_list)
            
            # Train policy with multiple steps of gradient descent
            for i in range(train_pi_iters):
                pi_optimizer.zero_grad()
                # NEED TO AVERAGE THESE HERE IN THIS LOOP
                loss_pi_avg = Variable(torch.zeros(1), requires_grad=True)
                pi_info_avg = dict()
                for agent in agent_type_list:
                    data = agent.buffer.get()
                    loss_pi, pi_info = compute_loss_pi(data, agent.brain)

                    loss_pi_avg += loss_pi

                    if not pi_info_avg:
                        pi_info_avg = pi_info.copy()
                    else: 
                        for k in pi_info.keys():
                            pi_info_avg[k] += pi_info[k]

                kl = mpi_avg(pi_info_avg['kl'])
                if kl > 1.5 * target_kl:
                    #logger.log('Early stopping at step %d due to reaching max kl.'%i)
                    break

                loss_pi_avg /= len(agent_type_list)
                loss_pi_avg.backward()
                # TODO: Get the average grads to refer to the correct neural network. 
                mpi_avg_grads(ac.pi)    # average grads across MPI processes
                pi_optimizer.step()

            #logger.store(StopIter=i)

            # Value function learning
            for i in range(train_v_iters):
                if agent_type_list[0].isByzantine: 
                    env.byz_v_function_optimizer.zero_grad()
                else: 
                    env.honest_v_function_optimizer.zero_grad()
                loss_v_avg = Variable(torch.zeros(1), requires_grad=True)
                for agent in agent_type_list:
                    data = agent.buffer.get()
                    if agent.isByzantine:
                        loss_v = compute_loss_v(data, env.byz_v_function)
                    else: 
                        loss_v = compute_loss_v(data, env.honest_v_function)
                    loss_v_avg+= loss_v
                    # at this point reset the buffer!!
                    agent.buffer.reset()
                loss_v_avg /= len(agent_type_list)

                loss_v_avg.backward()

                if agent_type_list[0].isByzantine: 
                    mpi_avg_grads(env.byz_v_function)
                    env.byz_v_function_optimizer.step()
                else: 
                    mpi_avg_grads(env.honest_v_function)
                    env.honest_v_function_optimizer.step()

                #mpi_avg_grads(ac.v)    # average grads across MPI processes
                #vf_optimizer.step()

            # Log changes from update
            kl, ent, cf = pi_info_avg['kl'], pi_info_old_avg['ent'], pi_info_avg['cf']
            '''logger.store(LossPi=pi_l_old_avg, LossV=v_l_old_avg,
                        KL=kl, Entropy=ent, ClipFrac=cf,
                        DeltaLossPi=(loss_pi_avg.item() - pi_l_old_avg),
                        DeltaLossV=(loss_v_avg.item() - v_l_old_avg))'''

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch): # will keep looping and even restarting the environment until the end here. 
            
            '''' not sure if I want the neural networks here in ppo. 
            no I want the updates to happen within the agents themselves. '''
            end_sim = env.step(ep_len, t) # episode length and then the total number of steps in the buffer. 
            
            #a, v, logp = env.step(torch.as_tensor(o, dtype=torch.float32))
            # action, value calcs, log probs. 
            #next_o, r, d, _ = env.step(a) # reward and indicator for died. 
            #ep_ret += r
            ep_len += 1

            # save and log
            #buf.store(o, a, r, v, logp)
            # TODO: figure out how to get the logger working. 
            #logger.store(VVals=v)
            
            # Update obs (critical!)
            #o = next_o
            #terminal = d or timeout
            #timeout = ep_len == max_ep_len
            #epoch_ended = t==local_steps_per_epoch-1 

            ''' I need to have the done signal only be true if all the agents have committed. 
            when a single agent commmits I need to run everything below for its own buffer. 
            but then I dont want to reset until all of them have done their things. keep 
            writing to the buffer with blanks until this happens. 
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v) # tie off this path no matter what
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len) '''

            if end_sim:
                o, ep_ret, ep_len = env.reset(), 0, 0 # reset the environment

        # TODO: get model save working. 
        # Save model
        #if (epoch % save_freq == 0) or (epoch == epochs-1):
        #    logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        # TODO: get the logger to work
        '''logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()'''

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
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)'''