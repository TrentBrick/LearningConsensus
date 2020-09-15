from ppo_code_gym.buffer import MultiAgentPPOBuffer
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import ppo_code_gym.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from ppo_code_gym.buffer import MultiAgentPPOBuffer


def ppo(env_fn, params, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
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
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(output_dir="/Users/yash/Documents/consensus/experiments/exp70-syncBA-4RoundFull-NoEquiv")
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    print(obs_dim)
    print(act_dim)
    print(env.observation_space)
    print(env.action_space)
    # Create actor-critic module
    # ac = torch.load('/tmp/experiments/exp46-syncBA-3Round-equiv/pyt_save/model.pt')
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = MultiAgentPPOBuffer(obs_dim, 1, local_steps_per_epoch, params['num_byzantine'])

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
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
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

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
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o_list, honest_ep_ret, byzantine_ep_ret, ep_len = env.reset(), 0, 0, 0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs): 
        round_len = 1
        curr_ep_trajectory_log = []
        prev_ep_trajectory_log = []
        single_run_trajectory_log = setup_trajectory_log(env.allAgents)
        sims=0
        honest_wins = 0
        byzantine_wins = 0
        rounds = 0
        safetyViolations = 0
        same_action = 0
        byzantine_action_dic = dict()
        for t in range(local_steps_per_epoch):
            actions_list = []
            v_list = []
            logp_list = []
            for i, agent in enumerate(env.agents):
                if agent.committed_value != params['null_message_val']:
                    a, logp, v = agent.actionIndex, None, None
                else:
                    a, v, logp = ac.step(torch.as_tensor(o_list[i], dtype=torch.float32))

                actions_list.append(a)
                v_list.append(v)
                logp_list.append(logp)

            #Store new values in buffer
            for ind, agent in enumerate(env.agents):
                if not agent.isByzantine:
                    continue
                agentActionString =  agent.actionSpace[actions_list[ind]]
                if 'commit' in agentActionString:
                    agent.committed_value = int(agentActionString.split('_')[1])
                
                if agent.committed_value == params['null_message_val']:
                    buf.store(0, o_list[ind], actions_list[ind], v_list[ind], logp_list[ind])

                elif agent.committed_value != params['null_message_val'] and len(agent.last_action_etc.keys()) == 0:
                    pass 
            
            next_o, r_list, d_list, info_n_list, sim_done, safety_violation = env.step(actions_list, v_list, logp_list, round_len)
            if round_len == 2:
                if env.byzantine_agents[0].prevActionString == env.byzantine_agents[0].actionString:
                    same_action+=1

            ep_len += 1

            

            #Log in trajectory
            for agent in env.allAgents:
                single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentId)].append((agent.state, agent.actionString, agent.status_values, agent.statusValue, agent.proposeValue))
           
           ## Print out safety violation
            if safety_violation:
                safetyViolations+=1
                

            for ind, agent in enumerate(env.agents):
                if agent.actionString in byzantine_action_dic:
                    byzantine_action_dic[agent.actionString]+=1
                else:
                    byzantine_action_dic[agent.actionString] = 1
                byzantine_ep_ret += r_list[ind]
                buf.store_reward(0, agent.reward)

            
            # Get v
            v=0
            for val in v_list:
                if val is not None:
                    v+=val
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o_list = next_o

            timeout = ep_len == max_ep_len
            terminal = sim_done or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                sims+=1
                allCorrectCommit = True
                comm_values = []

                for agent in env.honest_agents:
                    comm_values.append(agent.committed_value)

                if sim_done and round_len <= 4:
                    honest_wins+=1
                if sim_done and round_len > 4:
                    byzantine_wins+=1
                # if len(set(comm_values)) is 1:
                #     honest_wins+=1
                # else:
                #     byzantine_wins+=1

                if all(d_list):
                    single_run_trajectory_log['allCommit']+=1

                for ind, agent in enumerate(env.agents):
                    #If sim done because of reaching max round length, then record last action
                    if len(agent.last_action_etc.keys())==0:
                        agent.last_action_etc['obs'] = agent.state
                        agent.last_action_etc['act'] = actions_list[ind]
                        agent.last_action_etc['val'] = v_list[ind]
                        agent.last_action_etc['logp'] = logp_list[ind]

                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                v = 0
                if timeout or epoch_ended:
                    for ind, agent in enumerate(env.agents):
                        _, curr_v, _ = ac.step(torch.as_tensor(o_list[i], dtype=torch.float32))
                        v += curr_v
                buf.finish_sim(env.byzantine_agents)
                
                if terminal:
                    logger.store(EpRet=byzantine_ep_ret, ByzantineEpRet = byzantine_ep_ret, EpLen=ep_len)
                o_list, byzantine_ep_ret, ep_len = env.reset(), 0, 0
                rounds+= round_len
                round_len = 0
                prev_ep_trajectory_log = curr_ep_trajectory_log[:]
                curr_ep_trajectory_log.append(single_run_trajectory_log)
                single_run_trajectory_log = setup_trajectory_log(env.allAgents)

            round_len += 1

        # Print model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            print('=============================')
            print('second to last trajectory from this epoch:')
            for k, v in prev_ep_trajectory_log[-1].items():
                print(k, v)
                print('---------')
            print("number of sims: ", sims)
            print('---------')
            print("action dic:")
            for k, v in byzantine_action_dic.items():
                print(k, v)
                print('---------')
            print('=============================')
        # Save model
        if epoch == epochs-1:
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('Sims', sims)
        logger.log_tabular('Same Action', same_action)
        logger.log_tabular('ByzantineWinPercentage', byzantine_wins/sims)
        logger.log_tabular('HonestWinPercentage', honest_wins/sims)
        logger.log_tabular('AverageRounds', rounds/sims)
        logger.log_tabular('SafetyViolations', safetyViolations)
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
        logger.dump_tabular()

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

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)


def setup_trajectory_log(agent_list):
    single_run_trajectory_log = dict()
    single_run_trajectory_log['commitCount'] = 0
    single_run_trajectory_log['allCommit']=0
    for agent in agent_list:
        single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentId)] = []
    return single_run_trajectory_log