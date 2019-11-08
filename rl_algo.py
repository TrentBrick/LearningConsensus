import torch 
import numpy as np
#from environment_and_agent_utils import toOneHotState, toOneHotActions
#first going to implement vpg (https://spinningup.openai.com/en/latest/algorithms/vpg.html#documentation)
send_all_first_round_reward = 0.3
additional_round_penalty = -0.03
commit_to_majority = 0.5

def vpg(curr_ep_trajectory_logs, toOneHotState, toOneHotActions, adv_honest_nets=None, adv_byz_nets=None, use_vpg=False ):#, honest_action_to_ind, byz_action_to_ind ):
    # for byzantine and honest separately (need to sum over the different honest agents also):

    # for advantage this is discounted infinite. otherwise it is just the reward. 
    # compute advantage estimation
    # estimate policy gradient
    # policy update

    # curr_ep_trajectory_logs has a list of dictionaries, each dict is
    # separated out by the agent or reward and tuples of: (round_counter, state, action, action_logprob) 

    both_parties_adv_losses = []
    losses = [] # will store honest and then byz losses. 
    for isByz in [False, True]: # iterating through the honest and byzantine parties
        key_prefix = 'Byz-'+str(isByz)
        reward_ind = int(isByz) # if is byzantine, this is a 1, which is the index in the rewards for byzantine. 
        
        num_rounds = 0 # counts up the rounds and the number of trajectories. 
        logp_rewards_sum = 0
        num_trajectories = 0 # want to count this separately for honest and byzantine
        adv_losses = [0,0]
        for trajectory_iter in curr_ep_trajectory_logs: # going through each of the trajectories 
            
            termination_reward = trajectory_iter['reward'][reward_ind]

            for key, trajectory_rounds in trajectory_iter.items(): # going through the keys and their list of state, action, action prob pairs

                if key_prefix in key: #only getting byzantine or honest
                    
                    rewards_to_go = termination_reward

                    for round_ind in reversed(range(len(trajectory_rounds))): 
                        roundd = trajectory_rounds[round_ind]
                        agent_action_ind = roundd[4]
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
                        #if sum(agent_round_state) / len(agent_round_state) != 0.5:
                        
                        # if it is exactly in the middle then dont check or reward.
                        majority_value = int((sum(agent_round_state) / len(agent_round_state))+0.5)
                        if not isByz and 'commit' in agent_round_action and majority_value == int(agent_round_action.split('_')[1]) :
                            #print('committted to majority!!!!!!', majority_value, agent_round_state, agent_round_action)
                            rewards_to_go += commit_to_majority

                        #penalty for every additional round length: 
                        if not isByz:
                            rewards_to_go += additional_round_penalty
                        elif isByz: 
                            rewards_to_go -= additional_round_penalty

                        ### Finished computing the rewards to go. 
                        ### Computing the values and for the advantage function

                        # getting the values. v then q. 
                        if use_vpg:
                            if not isByz: 
                                adv_nets = adv_honest_nets
                                #agent_action_ind = honest_action_to_ind[agent_round_action] # need this to convert it into a onehot for the network
                            else: 
                                adv_nets = adv_byz_nets
                                #agent_action_ind = byz_action_to_ind[agent_round_action]

                            adv_preds = [] # v and then q
                            #make the state onehot
                            #print('action then state', agent_round_action, agent_round_state)
                            oh_state= toOneHotState(agent_round_state)
                            oh_action = toOneHotActions(isByz, agent_action_ind)
                            oh_action_state = torch.cat( (oh_action, oh_state),dim=0)
                            #print('checking onehotter', oh_action_state, agent_round_action, agent_action_ind)
                            #print('concatenated action and state', oh_action_state) 

                            for ind, net in enumerate(adv_nets): 
                                #print(ind)
                                if ind == 0: # this is the v function
                                    # could this be parallelized much more efficiently? compute once and then store?? 
                                    adv_pred = net(oh_state)
                                else: # this is the q function
                                    adv_pred = net(oh_action_state)

                                adv_losses[ind] += (adv_pred - rewards_to_go)**2

                                adv_preds.append(adv_pred.detach()) # detach for the other loss. just want the scalars. 

                                #print('this sshould still have tensors on it', adv_losses)

                        # DO I NEED TO DETACH THE PREDICTIONS HERE?? 
                        #print('av preds maybe detach', adv_preds)
                        if use_vpg: 
                            logp_rewards_sum += log_prob * (adv_preds[1] - adv_preds[0]) # Q - V advantage function
                        else: 
                            logp_rewards_sum += log_prob * rewards_to_go

                        num_rounds += 1

            num_trajectories +=1

        if adv_losses!=[0,0]: # there have been no updates because there is no byzantine. 
            adv_losses = np.asarray(adv_losses)/(num_rounds) # num rounds will also include the number of trajectories. 
            both_parties_adv_losses += list(adv_losses) # honest and then byzantine. 
            
        logp_rewards_sum /= num_trajectories
        logp_rewards_sum *= -1 # so that it is gradient ascent!
        #print('appending loss')
        losses.append(logp_rewards_sum)
        #print('just appended losses:', losses)

    if use_vpg:
        return losses, both_parties_adv_losses
    else: 
        return losses
