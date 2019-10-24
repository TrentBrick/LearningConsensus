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

    curr_ep = starting_ep
    curr_temperature=starting_temp
    temperature_tracker = []

    honest_wins_total = []
    total_honest_rewards = []
    
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

        curr_ep_trajectory_logs = []

        satisfied_constraints = []
        ep_honest_reward = 0

        hit_max_round_len = 0
        avg_round_len = 0

        for iter_in_ep in range(iters_per_epoch):
            #run the environment. 

            single_run_trajectory_log = dict()

            #initialize the values and which agents are byzantine. 
            # agent_list is all agents, honest and byzantine are subsets. 
            agent_list, honest_list, byzantine_list = initStatesandAgents()

            round_counter = 0
            #until honest parties commit values (simulation terminates)
            while not honestPartiesCommit(honest_list):
                # choose new actions: 
                for agent in agent_list: 

                    if type(agent.committed_value) is int:      # dont change to True! Either it is False or a real value. 
                        action, action_logprob = agent.action, None
                    else:
                        if round_counter>max_round_len: # force the honest agents to commit to a value. 
                            action, action_logprob = agent.chooseAction(curr_temperature, forceCommit=True)
                        else: 
                            action, action_logprob = agent.chooseAction(curr_temperature)
                    
                    try: 
                        single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentID)].append( (round_counter, agent.state, action, action_logprob ))
                    except: 
                        single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentID)] = [ (round_counter, agent.state, action, action_logprob) ]


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
            reward, satisfied_constraints_iter = giveReward(honest_list, single_run_trajectory_log)

            ep_honest_reward += reward[0]
            #print('reward for iter:', reward)

            # storing in loggers
            satisfied_constraints.append(satisfied_constraints_iter)
            single_run_trajectory_log['reward'] = reward
            curr_ep_trajectory_logs.append(single_run_trajectory_log)

        total_trajectory_logs.append(curr_ep_trajectory_logs[-1] )

        losses = rl_algo(curr_ep_trajectory_logs)
        honest_loss = losses[0]
        byz_loss = losses[1]

        honest_loss.backward()
        if num_byzantine!=0:
            byz_loss.backward()

        honest_optimizer.step()
        if num_byzantine!=0:
            byz_optimizer.step()

        #compute the loss using the RL algorithm
        honest_victory = [ 1 if s==True else 0 for s in satisfied_constraints  ]
        #honest_wins_total += honest_victory
        #byz_rewards = sum([ s[1] for s in satisfied_constraints ])
        honest_wins_total.append(sum(honest_victory)/iters_per_epoch)
        total_honest_rewards.append(losses[0])
        

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
            print('Epoch Sum of Honest Rewards', ep_honest_reward/iters_per_epoch)
            print( 'Honest wins this epoch %', sum(honest_victory)/iters_per_epoch, '=============')
            #print( 'cum sum of honest wins', sum(honest_wins_total)*iters_per_epoch, '=============')
            #print('as a percentage of all trajectories:', (sum(honest_wins_total)*iters_per_epoch)/ (curr_ep*iters_per_epoch))
            print('Current Epoch is: ', curr_ep)
            print('Current Temperature is:' , curr_temperature, '=======')
            print('Losses from the Epoch', losses)
            print('=============================')
            print('=============================')
            #print useful information. 

        curr_ep += 1

    # plot the change in temperature over time. 
    # plot average honest dddwin rate over time. 
    save_labels = ['honest_wins', 'temperature', 'average_loss']
    for to_plot, label in zip([honest_wins_total, temperature_tracker, total_honest_rewards],save_labels):
        savePlot(to_plot, label)

    pickle.dump(total_trajectory_logs, open(directory+'trajectory_logs-'+experiment_name+'.pickle', 'wb'))

# if the policy is better then save it. is overfitting a problem in RL? 

if __name__=='__main__':
    main()