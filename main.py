''' Script that is called to allow for code to be executed.  '''
import torch 
from config import *
from environment_utils import *
import time 

def main():

    model.zero_grad()

    init_e = curr_ep
    
    #first_ep_first_batch_only=None

    total_trajectory_logs = []

    while curr_ep < (epochs+init_e):  
        print('Epoch', curr_ep)

        model.zero_grad()

        curr_ep_trajectory_logs = []

        ep_rewards = []

        for iter_in_ep in range(iters_per_epoch):
            #run the environment. 

            single_run_trajectory_log = []

            #initialize the values and which agents are byzantine
            agent_list, honest_list, byzantine_list = initStatesandAgents()

            #until honest parties commit values (simulation terminates)
            while not honestPartiesCommit(honest_list):

                round = []

                # choose new actions: 
                for agent in agent_list: 
                    state, action = agent.chooseAction()
                    round.append( (state, action) )

                # log the current state and action
                single_run_trajectory_log.append(round) # list of list of vectors

                # resolve the new states: 
                for agent in agent_list: 
                    updateStates()

                # keep making more actions, storing all 
                # of them along with the states and rewards

            # upon termination, calculate the terminal reward:
            # currently just checking if the agents satisfied consistency and validity
            # recieves a tuple of the form honest reward, byzantine reward
            reward = giveReward(honest_list)

            # storing in loggers
            ep_rewards.append(reward)
            single_run_trajectory_log.append(reward)
            curr_ep_trajectory_logs.append(single_run_trajectory_log)

        total_trajectory_logs.append(curr_ep_trajectory_logs)

        #compute the loss using the RL algorithm
        honest_reward = sum([ s[0] for s in ep_rewards ])
        byz_reward = sum([ s[1] for s in ep_rewards ])
        
        '''loss = 

        loss.backward()

        optimizer.step()'''

        # get all of the relevant metrics. eg. loss.item()

    if (curr_ep % print_every == 0):
        print('Current Epoch is: ', curr_ep)
        #print useful information. 

# if the policy is better then save it. is overfitting a problem in RL? 

if __name__=='__main__':
    main()