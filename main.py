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
    
    #first_ep_first_batch_only=None

    total_trajectory_logs = []

    while curr_ep < (epochs+starting_ep):  
        print('Epoch', curr_ep)

        if curr_temperature>temp_fix_point:
            curr_temperature = curr_temperature*temp_anneal # anneal the temperature for selecting actions over time. 
        temperature_tracker.append(curr_temperature)

        honest_policy.zero_grad()
        byz_policy.zero_grad()

        curr_ep_trajectory_logs = []

        ep_rewards = []

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
                        action, action_logprob = agent.action, 0
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
                    print('too many rounds!!!', round_counter)
                    #print(single_run_trajectory_log)

                round_counter+=1

            #print('single trajectory over:', single_run_trajectory_log)

            # upon termination, calculate the terminal reward:
            # currently just checking if the agents satisfied consistency and validity
            # recieves a tuple of the form honest reward, byzantine reward
            reward = giveReward(honest_list)

            #print('reward for iter:', reward)

            # storing in loggers
            ep_rewards.append(reward)
            single_run_trajectory_log['reward'] = reward
            curr_ep_trajectory_logs.append(single_run_trajectory_log)

        total_trajectory_logs.append(curr_ep_trajectory_logs )

        #compute the loss using the RL algorithm
        honest_rewards = [ 1 if s[0]==1 else 0 for s in ep_rewards  ]

        #honest_wins_total += honest_rewards
        honest_wins_total.append(sum(honest_rewards)/iters_per_epoch)
        print( 'honest wins this epoch', sum(honest_rewards), '=============')
        print( 'honest wins this epoch %', sum(honest_rewards)/iters_per_epoch, '=============')
        print( 'cum sum of honest wins', sum(honest_wins_total)*iters_per_epoch, '=============')
        print('as a percentage of all trajectories:', (sum(honest_wins_total)*iters_per_epoch)/ (curr_ep*iters_per_epoch))

        #byz_rewards = sum([ s[1] for s in ep_rewards ])
        
        losses = rl_algo(curr_ep_trajectory_logs)
        honest_loss = losses[0]
        byz_loss = losses[1]

        honest_loss.backward()
        if num_byzantine!=0:
            byz_loss.backward()

        honest_optimizer.step()
        if num_byzantine!=0:
            byz_optimizer.step()

        # get all of the relevant metrics. eg. loss.item()

        if (curr_ep % print_every == 0):
            print('=============================')
            print('Current Epoch is: ', curr_ep)
            print('Current Temperature is:' , curr_temperature, '=======')
            print('last trajectory from this epoch:')
            print(curr_ep_trajectory_logs[-1])
            print('very first!')
            print(curr_ep_trajectory_logs[0])
            print('=============================')
            print('=============================')
            #print useful information. 

        curr_ep += 1

    # plot the change in temperature over time. 
    # plot average honest win rate over time. 
    plt.plot(range(len(honest_wins_total)), honest_wins_total, label='honest_win_%')
    plt.plot(range(len(temperature_tracker)), temperature_tracker, label='temperature')
    plt.xlabel('epochs')
    plt.ylabel('\% honest wins in the epoch')
    plt.title(str(iters_per_epoch)+' iters per epoch')
    plt.legend()
    plt.gcf().savefig(directory+'honest_wins-'+experiment_name+'.png', dpi=200)
    
    pickle.dump(total_trajectory_logs, open(directory+'trajectory_logs-'+experiment_name+'.pickle', 'wb'))

# if the policy is better then save it. is overfitting a problem in RL? 

if __name__=='__main__':
    main()