''' Script that is called to allow for code to be executed.  '''
import torch 
from config import *
from environment_and_agent_utils import *
import time 
import matplotlib.pyplot as plt
import pickle

def main():

    honest_policy.zero_grad()
    byz_policy.zero_grad()

    for net in [honest_v_function, honest_q_function, byz_v_function, byz_q_function]:
        net.zero_grad()
        net.train()

    curr_ep = starting_ep
    curr_temperature=starting_temp
    temperature_tracker = []

    honest_wins_total = []
    honest_adv_loss_v = []
    honest_adv_loss_q = []
    honest_losses = []
    byzantine_losses = []
    honest_rewards = []
    byzantine_rewards = []
    
    #first_ep_first_batch_only=None

    total_trajectory_logs = []

    while curr_ep < (epochs+starting_ep):  
        print('Epoch', curr_ep)

        if use_heat_jumps:
            curr_temperature = curr_temperature*temp_anneal # anneal the temperature for selecting actions over time. 
            if curr_temperature<temp_fix_point: # this will bump up the temperature again after having annealed it. 
                curr_temperature = starting_temp
        else: 
            if curr_temperature>temp_fix_point: # only decrease temp if it is above threshold
                curr_temperature = curr_temperature*temp_anneal

        temperature_tracker.append(curr_temperature)
        
        honest_policy.zero_grad()
        byz_policy.zero_grad()

        for net in [honest_v_function, honest_q_function, byz_v_function, byz_q_function]:
            net.zero_grad()

        curr_ep_trajectory_logs = []

        satisfied_constraints = []
        epoch_honest_reward = 0
        epoch_byz_reward = 0

        hit_max_round_len = 0
        avg_round_len = 0

        for iter_in_ep in range(iters_per_epoch):
            #run the environment. 

            single_run_trajectory_log = dict()

            #initialize the values and which agents are byzantine. 
            # agent_list is all agents, honest and byzantine are subsets. 
            agent_list, honest_list, byzantine_list, byzantine_inds = initStatesandAgents()

            # need to update the byzantine action to ind dictionary to account for the current byzantine index
            #byz_action_space = getActionSpace(True, byzantine_inds=byzantine_inds, can_send_either_value=honest_can_send_either_value)
            #byz_action_to_ind = {a:ind for ind, a in enumerate(byz_action_space)}

            round_counter = 0
            #until honest parties commit values (simulation terminates)
            while not honestPartiesCommit(honest_list):
                # choose new actions: 
                for agent in agent_list: 

                    if type(agent.committed_value) is int:      # dont change to True! Either it is False or a real value. 
                        action, action_logprob = agent.action, None
                    else:
                        if round_counter>max_round_len: # force the honest agents to commit to a value. 
                            action, action_logprob, action_ind = agent.chooseAction(curr_temperature, forceCommit=True)
                        else: 
                            action, action_logprob, action_ind = agent.chooseAction(curr_temperature)
                    
                    try: 
                        single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentID)].append( (round_counter, agent.state, action, action_logprob, action_ind ))
                    except: 
                        single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentID)] = [ (round_counter, agent.state, action, action_logprob, action_ind) ]


                # log the current state and action

                # resolve the new states: 
                for agent in agent_list: 
                    updateStates(agent_list)

                # keep making more actions, storing all 
                # of them along with the states and rewards

                if round_counter> max_round_len:
                    hit_max_round_len +=1
                    #print('too many rounds!!!', round_counter)
                    #print(single_run_trajectory_log)

                round_counter+=1

            avg_round_len += round_counter

            #print('single trajectory over:', single_run_trajectory_log)

            # upon termination, calculate the terminal reward:
            # currently just checking if the agents satisfied consistency and validity
            # recieves a tuple of the form honest reward, byzantine reward
            reward, satisfied_constraints_this_iter = giveReward(honest_list, single_run_trajectory_log)

            epoch_honest_reward += reward[0]
            epoch_byz_reward += reward[1]
            #print('reward for iter:', reward)

            # storing in loggers
            satisfied_constraints.append(satisfied_constraints_this_iter)
            single_run_trajectory_log['reward'] = reward
            curr_ep_trajectory_logs.append(single_run_trajectory_log)

        total_trajectory_logs.append(curr_ep_trajectory_logs[-1] )

        losses, adv_losses = rl_algo(curr_ep_trajectory_logs, [honest_v_function, honest_q_function],
        [byz_v_function, byz_q_function], toOneHotState, 
        toOneHotActions)#, honest_action_to_ind, byz_action_to_ind)
        #print("adv losses, should be flattened", adv_losses)
        honest_loss = losses[0] # store like this so they are interpretable and can print later if want to. 
        byz_loss = losses[1]
        honest_losses.append(honest_loss)
        byzantine_losses.append(byz_loss)

        honest_rewards.append(epoch_honest_reward)
        byzantine_rewards.append(epoch_byz_reward)

        honest_adv_loss_v.append(adv_losses[0])
        honest_adv_loss_q.append(adv_losses[1])

        honest_loss.backward()
        honest_optimizer.step()

        if num_byzantine!=0:
            byz_loss.backward()
            byz_optimizer.step()

        # update the advantage functions: 
        for ind, adv_loss in enumerate(adv_losses):
            #print('this adv loss is:', adv_loss)
            adv_loss.backward()
            adv_optimizers[ind].step()

        #compute store the losses and other metrics: 
        honest_victory = [ 1 if s==True else 0 for s in satisfied_constraints  ]
        honest_losses.append(honest_loss)
        byzantine_losses.append(byz_loss)

        #honest_wins_total += honest_victory
        #byz_rewards = sum([ s[1] for s in satisfied_constraints ])
        honest_wins_total.append(sum(honest_victory)/iters_per_epoch)
        
        # get all of the relevant metrics. eg. loss.item()

        if (curr_ep % print_every == 0):
            print('=============================')
            
            print('last trajectory from this epoch:')
            for k, v in curr_ep_trajectory_logs[-1].items():
                print(k, v)
                print('---------')
            print('Hit the max round length %:', hit_max_round_len/iters_per_epoch)
            #print( 'Honest wins this epoch', sum(honest_victory), '=============')
            print('Average round length', avg_round_len/iters_per_epoch)
            print('Epoch Sum of Honest Rewards', epoch_honest_reward/iters_per_epoch)
            print( 'Honest wins this epoch %', sum(honest_victory)/iters_per_epoch, '=============')
            print('Honest losses', honest_loss, 'Byz losses', byz_loss)
            #print( 'cum sum of honest wins', sum(honest_wins_total)*iters_per_epoch, '=============')
            #print('as a percentage of all trajectories:', (sum(honest_wins_total)*iters_per_epoch)/ (curr_ep*iters_per_epoch))
            print('Current Epoch is: ', curr_ep)
            print('Current Temperature is:' , curr_temperature, '=======')
            print('Advantage Losses', adv_losses)
            print('Losses from the Epoch', losses)
            print('=============================')
            print('=============================')
            #print useful information. 

        curr_ep += 1

    # plot the change in temperature over time. 
    # plot average honest dddwin rate over time. 
    save_labels = ['honest_wins', 'temperature', 'honest_loss', 'byz_loss', 
    'honest_adv_loss_v', 'honest_adv_loss_q', 'honest_rewards', 'byz_rewards']
    for to_plot, label in zip([honest_wins_total, temperature_tracker, 
    honest_losses, byzantine_losses, honest_adv_loss_v, 
    honest_adv_loss_q, honest_rewards, byzantine_rewards],save_labels):
        savePlot(to_plot, label)

    pickle.dump(total_trajectory_logs, open('runs/trajectory_logs-'+experiment_name+'.pickle', 'wb'))


    save_names = ['honest_policy', 'honest_policy_v', 'honest_policy_q', 
    'byz_policy', 'byz_policy_v', 'byz_policy_q']
    save_models = [honest_policy, honest_v_function, honest_q_function, 
    byz_policy, byz_v_function, byz_q_function]
    for m, n in zip(save_models, save_names):
        torch.save(m, 'saved_models/'+experiment_name+n+'.torch')
    
# if the policy is better then save it. is overfitting a problem in RL? 

if __name__=='__main__':
    main()