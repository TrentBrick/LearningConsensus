''' Script that is called to allow for code to be executed.  '''
import torch 
from config import *
from environment_and_agent_utils import *
import time 

def main():

    honest_policy.zero_grad()
    byz_policy.zero_grad()

    init_e = curr_ep
    curr_temperature=starting_temp
    
    #first_ep_first_batch_only=None

    total_trajectory_logs = []

    while curr_ep < (epochs+init_e):  
        print('Epoch', curr_ep)

        curr_temperature = curr_temperature*temp_anneal # anneal the temperature for selecting actions over time. 

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

                    if round_counter>max_round_len: # force the honest agents to commit to a value. 
                        state, action, action_logprob = agent.chooseAction(curr_temperature, forceCommit=True)
                    else: 
                        state, action, action_logprob = agent.chooseAction(curr_temperature)
                    try: 
                        single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentID)].append( (round_counter, state, action, action_logprob ))
                    except: 
                        single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentID)] = [ (round_counter, state, action, action_logprob) ]


                # log the current state and action

                # resolve the new states: 
                for agent in agent_list: 
                    updateStates(agent_list)

                # keep making more actions, storing all 
                # of them along with the states and rewards
                round_counter+=1

            print('single trajectory over:', single_run_trajectory_log)

            # upon termination, calculate the terminal reward:
            # currently just checking if the agents satisfied consistency and validity
            # recieves a tuple of the form honest reward, byzantine reward
            reward = giveReward(honest_list)

            print('reward for iter:', reward)

            # storing in loggers
            ep_rewards.append(reward)
            single_run_trajectory_log['reward'] = reward
            curr_ep_trajectory_logs.append(single_run_trajectory_log)

        total_trajectory_logs.append(curr_ep_trajectory_logs, )

        #compute the loss using the RL algorithm
        honest_reward = sum([ s[0] for s in ep_rewards ])
        byz_reward = sum([ s[1] for s in ep_rewards ])
        
        losses = rl_algo(curr_ep_trajectory_logs)
        honest_loss = losses[0]
        byz_loss = losses[1]

        honest_loss.backward()
        byz_loss.backward()

        honest_optimizer.step()
        byz_optimizer.step()

        # get all of the relevant metrics. eg. loss.item()

    if (curr_ep % print_every == 0):
        print('Current Epoch is: ', curr_ep)
        #print useful information. 

# if the policy is better then save it. is overfitting a problem in RL? 

if __name__=='__main__':
    main()