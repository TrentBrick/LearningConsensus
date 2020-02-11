import numpy as np 
import torch
from environment_and_agent_utils import getActionSpace, giveReward, honestPartiesCommit, updateStates, initStatesandAgents
# Functions used by the main.py training loop

def run_iters(params, honest_curr_temperature, 
        byz_curr_temperature, honest_policy, byz_policy, oneHotStateMapper, device):
    curr_ep_trajectory_logs = []
    satisfied_constraints = []
    epoch_honest_reward = 0
    epoch_byz_reward = 0

    hit_max_round_len = 0
    avg_round_len = 0

    for round_in_ep in range(params['rounds_per_epoch']):
        #run the environment. 

        single_run_trajectory_log = dict()

        #initialize the values and which agents are byzantine. 
        # agent_list is all agents, honest and byzantine are subsets. 
        agent_list, honest_list, byzantine_list = initStatesandAgents(params, honest_policy, byz_policy)

        round_counter = 0
        #until honest parties commit values (simulation terminates)
        while not honestPartiesCommit(honest_list):
            # choose new actions: 
            for agent in agent_list: 
                if agent.isByzantine: 
                    curr_temperature = byz_curr_temperature
                else: 
                    curr_temperature = honest_curr_temperature

                if type(agent.committed_value) is int:      # dont change to True! Either it is False or a real value. 
                    action, action_logprob = agent.action, None
                else:
                    if round_counter>params['max_round_len']: # force the honest agents to commit to a value. 
                        action, action_logprob = agent.chooseAction(oneHotStateMapper, curr_temperature, device, forceCommit=True)
                    else: 
                        action, action_logprob = agent.chooseAction(oneHotStateMapper, curr_temperature, device)
                
                # log the current state and action
                try: 
                    single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentID)].append( (round_counter, agent.state, action, action_logprob ))
                except: 
                    single_run_trajectory_log['Byz-'+str(agent.isByzantine)+'_agent-'+str(agent.agentID)] = [ (round_counter, agent.state, action, action_logprob) ]

            # resolve the new states: 
            #for agent in agent_list: 
            updateStates(params, agent_list)

            # keep making more actions, storing all 
            # of them along with the states and rewards
            if round_counter> params['max_round_len']:
                hit_max_round_len +=1

            round_counter+=1

        avg_round_len += round_counter

        # upon termination, calculate the terminal reward:
        # currently just checking if the agents satisfied consistency and validity
        # recieves a tuple of the form honest reward, byzantine reward
        reward, satisfied_constraints_this_iter = giveReward(params, honest_list, single_run_trajectory_log)

        epoch_honest_reward += reward[0]
        epoch_byz_reward += reward[1]
        #print('reward for iter:', reward)

        # storing in loggers
        satisfied_constraints.append(satisfied_constraints_this_iter)
        single_run_trajectory_log['reward'] = reward
        curr_ep_trajectory_logs.append(single_run_trajectory_log)

    #total_trajectory_logs.append(curr_ep_trajectory_logs[-1] )
    
    return curr_ep_trajectory_logs, satisfied_constraints, epoch_honest_reward, epoch_byz_reward, hit_max_round_len, avg_round_len

def temp_annealer(params, honest_curr_temperature, byz_curr_temperature):
    if params['use_heat_jumps']:
        honest_curr_temperature = honest_curr_temperature*params['temp_anneal'] # anneal the temperature for selecting actions over time. 
        if honest_curr_temperature<params['temp_fix_point']: # this will bump up the temperature again after having annealed it. 
            honest_curr_temperature = params['starting_temp']
        if byz_curr_temperature<params['temp_fix_point']:
            byz_curr_temperature=params['starting_temp']
    else: 
        if honest_curr_temperature>params['temp_fix_point']: # only decrease temp if it is above threshold
            honest_curr_temperature = honest_curr_temperature*params['temp_anneal']
        if byz_curr_temperature>params['temp_fix_point']:
            byz_curr_temperature=byz_curr_temperature*params['temp_anneal']
    return honest_curr_temperature, byz_curr_temperature

