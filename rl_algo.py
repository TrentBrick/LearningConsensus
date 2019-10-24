#first going to implement vpg (https://spinningup.openai.com/en/latest/algorithms/vpg.html#documentation)
send_all_first_round_reward = 0.3
additional_round_penalty = -0.03
commit_to_majority = 0.5

def vpg(curr_ep_trajectory_logs):
    # for byzantine and honest separately (need to sum over the different honest agents also):

    # compute the rewards to go
    # for advantage this is discounted infinite. otherwise it is just the reward. 
    # compute advantage estimation
    # estimate policy gradient
    # policy update

    # curr_ep_trajectory_logs has a list of dictionaries, each dict is
    # separated out by the agent or reward and tuples of: (round_counter, state, action, action_logprob) 

    
    losses = [] # will store honest and then byz losses. 

    for isByz in [False, True]: # iterating through the honest and byzantine parties
        key_prefix = 'Byz-'+str(isByz)
        reward_ind = int(isByz) # if is byzantine, this is a 1, which is the index in the rewards for byzantine. 
        
        logp_rewards_sum = 0
        num_trajectories = 0 # want to count this separately for honest and byzantine
        
        for trajectory_iter in curr_ep_trajectory_logs: # going through each of the trajectories 
            
            termination_reward = trajectory_iter['reward'][reward_ind]

            for key, trajectory_rounds in trajectory_iter.items(): # going through the keys and their list of state, action, action prob pairs

                if key_prefix in key: #only getting byzantine or honest
                    
                    rewards_to_go = termination_reward

                    for round_ind in reversed(range(len(trajectory_rounds))): 
                        roundd = trajectory_rounds[round_ind]
                        log_prob = roundd[3]
                        agent_round_state = roundd[1]
                        agent_round_action = roundd[2]
                        if log_prob is None: 
                            #print('log prob is none', log_prob, 'round ind is:', round_ind)
                            #print('trajectory rounds are: ', trajectory_rounds)
                            continue
                        # (round_counter, state, action, action_logprob)
                        # rewards to go are all the same!

                        #for byzantine need to ignore its very last action as the other agents already both decided to commit!
                        if isByz and round_ind >= len(trajectory_rounds)-1:
                            continue

                        # reward for not committing in the first round
                        if not isByz and round_ind == 0 and 'send_to_all-' in agent_round_action:
                            rewards_to_go += send_all_first_round_reward

                        # if this agent committed to the majority value, reward them
                        majority_value = int((sum(agent_round_state) / len(agent_round_state))+0.5)
                        if not isByz and 'commit' in agent_round_action and  majority_value == int(agent_round_action.split('_')[1]) :
                            #print('committted to majority!!!!!!', majority_value, agent_round_state, agent_round_action)
                            rewards_to_go += commit_to_majority

                        #penalty for every additional round length: 
                        if not isByz:
                            rewards_to_go += additional_round_penalty
                        
                        logp_rewards_sum += log_prob * rewards_to_go

            num_trajectories +=1

        logp_rewards_sum /= num_trajectories
        logp_rewards_sum *= -1 # so that it is gradient ascent!
        losses.append(logp_rewards_sum)

    return losses



                


