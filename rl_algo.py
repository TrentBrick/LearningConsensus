#first going to implement vpg (https://spinningup.openai.com/en/latest/algorithms/vpg.html#documentation)

def vpg(curr_ep_trajectory_logs):
    # for byzantine and honest separately (need to sum over the different honest agents also):

    # compute the rewards to go
    # for advantage this is discounted infinite. otherwise it is just the reward. 
    # compute advantage estimation
    # estimate policy gradient
    # policy update

    # curr_ep_trajectory_logs has a list of dictionaries, each dict is
    # separated out by the agent or reward and tuples of: (round_counter, state, action, action_logprob) 

    num_trajectories = len(curr_ep_trajectory_logs)
    losses = [] # will store honest and then byz losses. 

    for isByz in [False, True]: # iterating through the honest and byzantine parties
        key_prefix = 'Byz-'+str(isByz)
        reward_ind = int(isByz) # if is byzantine, this is a 1, which is the index in the rewards for byzantine. 
        
        logp_rewards_sum = 0
        
        for trajectory_iter in curr_ep_trajectory_logs: # going through each of the trajectories 
            
            reward = trajectory_iter['reward'][reward_ind]

            for key, trajectory_rounds in trajectory_iter.items(): # going through the keys and their list of state, action, action prob pairs

                if key_prefix in key: #only getting byzantine or honest
                    
                    for round_ind, roundd in enumerate(trajectory_rounds): 
                        # (round_counter, state, action, action_logprob)
                        # rewards to go are all the same!

                        #need to ignore the continued commit actions that happened after the very first commit.
                        if 'commit' in roundd[2]:
                            logp_rewards_sum += roundd[3] * reward
                            break # moves onto the next agent. 

                        #for byzantine need to ignore its very last action as the other agents already both decided to commit!
                        if isByz and round_ind >= len(trajectory_rounds)-1:
                            break

                        logp_rewards_sum += roundd[3] * reward

        logp_rewards_sum /= num_trajectories
        logp_rewards_sum *= -1 # so that it is gradient ascent!
        losses.append(logp_rewards_sum)

    return losses



                


