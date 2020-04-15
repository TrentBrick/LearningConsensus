import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import ppo_code_gym.core as core
from consensus_env import onehotter
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from ppo_code_gym.buffer import MultiAgentPPOBuffer


# TODO: make the hidden sizes updatable through run.py. 
def ppo(env_fn, params, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=.03,
        vf_lr=.03, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=5):
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
        honest_logger_kwargs (dict): Keyword args for Epochhonest_logger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up honest_logger and save configuration
    honest_logger = EpochLogger(**logger_kwargs)
    # honest_logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn
    #TODO: this only works for all honest agents. 
    obs_dim = env.observation_space[0].shape #Take first element b/c all of them have the same size
    act_dim = env.action_space[0].shape #Take first element b/c all have the same size

    # Create actor-critic module
    #TODO: using the first part
    # TODO: only works for all honest. 
    # box then discrete. What is .n here? 
    #print('makign the actor critics. ', actor_critic)
    ac = actor_critic(env.observation_space[0], env.action_space[0], **ac_kwargs)
    #print('made the actor critics. ')
    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    honest_logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_actions_per_epoch = steps_per_epoch
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    # buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    #TODO: change the diff of params term
    buf = MultiAgentPPOBuffer(obs_dim, 1, local_steps_per_epoch, params['num_agents']-params['num_byzantine'])
    # Set uxp function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        #TODO: do we need to make this a onehotter to fix the bug on line 210?
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        # print("logp: ", len(logp))
        # print("compute loss sizes of evertyhing", obs.shape, act.shape, adv.shape, logp_old.shape)
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
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    # honest_logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                honest_logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        # TODO: need this logger to correspond to our actual training system. 
        honest_logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        honest_logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o_list, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        sim_done = False
        epoch_done = False
        # epoch_rewards = []
        curr_ep_trajectory_log = []
        single_run_trajectory_log = setup_trajectory_log(env.agents)
        round_len = 1
        while not epoch_done:       
            actions_list = []
            v_list = []
            logp_list = []
            for i, agent in enumerate(env.agents):
                ## Only take a step in the neural network if not committed
                if type(agent.committed_value) is int:
                    a, logp, v = agent.actionIndex, None, None
                else:
                    a, v, logp = ac.step(torch.as_tensor(o_list[i], dtype=torch.float))

                actions_list.append(a)
                v_list.append(v)
                logp_list.append(logp)

            #Store the new values in the buffer
            for ind, agent in enumerate(env.agents):
                agentActionString =  agent.actionSpace[actions_list[ind]]
                if 'commit' in agentActionString:
                    agent.committed_value = int(agentActionString.split('_')[1])
                
                if type(agent.committed_value) is bool:
                    buf.store(ind, o_list[ind], actions_list[ind], v_list[ind], logp_list[ind])

                elif type(agent.committed_value) is int and len(agent.last_action_etc.keys()) == 0:
                    pass 
                    #We handle this in the below step function, before all states are updated

            # Update the environment for each agent and calculate rewards
            next_o_list, r_list, d_list, info_n_list, sim_done = env.step(actions_list, v_list, logp_list, round_len)
            ep_ret += sum(r_list)
            ep_len += 1

            for agent in env.agents:
                single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentId)].append((agent.state, agent.actionString))


            #Get v
            v=0
            for val in v_list:
                if val is not None:
                    v+=val
            honest_logger.store(VVals=v)
            
            #Update obs
            o_list = next_o_list

            #Store new reward values
            for ind, agent in enumerate(env.agents):
                if type(agent.committed_value) is bool:
                    buf.store_reward(ind, agent.reward)     

            if sim_done:
                # epoch_rewards.append(round(sum(r_list)/len(r_list), 4))
                for ind, agent in enumerate(env.agents):
                    #If sim done because of reaching max round length, then record last action
                    if len(agent.last_action_etc.keys())==0:
                        agent.last_action_etc['obs'] = agent.state
                        agent.last_action_etc['act'] = action_n[ind]
                        agent.last_action_etc['val'] = v_list[ind]
                        agent.last_action_etc['logp'] = logp_list[ind]

                buf.finish_sim(env.agents)
                o_list, ep_ret, ep_len = env.reset(), 0, 0
                round_len = 0
                curr_ep_trajectory_log.append(single_run_trajectory_log)
                single_run_trajectory_log = setup_trajectory_log(env.agents)
                if buf.ptr > local_actions_per_epoch:
                    epoch_done = True

            round_len+=1


        honest_logger.store(EpRet=ep_ret, EpLen=ep_len)

       # save and log



            
            # Update obs (critical!)

            # timeout = ep_len == max_ep_len
            # terminal = sim_done or timeout
            # epoch_ended = t==local_steps_per_epoch-1

            # if terminal or epoch_ended:
            #     if epoch_ended and not(terminal):
            #         print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
            #     # if trajectory didn't reach terminal state, bootstrap value target
            #     if timeout or epoch_ended:
            #         v = 0
            #         for ind, agent in enumerate(env.agents):
            #             _, curr_v, _ = ac.step(torch.as_tensor(o_list[i], dtype=torch.float))
            #             v+=curr_v

            #             # _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.uint8))
                    
            #     else:
            #         v = 0
            #     buf.finish_sim(env.agents)
            #     # buf.finish_path(v)
            #     if terminal:
            #         # only save EpRet / EpLen if trajectory finished
            #         honest_logger.store(EpRet=ep_ret, EpLen=ep_len)
            #     o_list, ep_ret, ep_len = env.reset(), 0, 0
            #     round_len=0



        # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs-1):
        #     honest_logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()


        ### Print out actions
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            print('=============================')
            print('last trajectory from this epoch:')
            for k, v in curr_ep_trajectory_log[-1].items():
                print(k, v)
                print('---------')
            print('=============================')
        # Log info about epoch
        honest_logger.log_tabular('Epoch', epoch)
        honest_logger.log_tabular('EpRet', with_min_and_max=True)
        honest_logger.log_tabular('EpLen', average_only=True)
        honest_logger.log_tabular('VVals', with_min_and_max=True)
        honest_logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
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

if __name__ == '__main__':
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

    from spinup.utils.run_utils import setup_honest_logger_kwargs
    honest_logger_kwargs = setup_honest_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        honest_logger_kwargs=honest_logger_kwargs)

def setup_trajectory_log(agent_list):
    single_run_trajectory_log = dict()
    for agent in agent_list:
        single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentId)] = []
    return single_run_trajectory_log