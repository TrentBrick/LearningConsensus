''' Script that is called to allow for code to be executed.  '''
import torch 
#from config import *
from environment_and_agent_utils import *
from rl_algo import vpg
from train_funcs import *
import time 
import matplotlib.pyplot as plt
import pickle
import datetime
import argparse
import itertools
from collections import OrderedDict

def main(params):

    ################## Initialize parameters #################
    ##Get activation function from string input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    iters_per_epoch = params['iters_per_epoch']

    # Set rl_algo
    if params['rl_algo_wanted']=='vpg':
        rl_algo = vpg

    date_time = str(datetime.datetime.now()).replace(' ', '_')
    # used to save the policy and its outputs. 
    experiment_name = params['exp_name']+"rand_seed-%s_scenario-%s_epochs-%s_iters_per_ep-%s_rl_algo-%s_time-%s" % (params['random_seed'], params['scenario'], 
    params['epochs'], iters_per_epoch, params['rl_algo_wanted'], date_time )

    activation = getActivation(params['activation'])
    output_activation = getActivation(params['output_activation'])
    
    #Get state_oh_size
    if params['use_PKI']: 
        state_oh_size = (len(params['commit_vals'])+1)*(params['num_agents']**2-params['num_agents']+1)
    else: 
        state_oh_size = (len(params['commit_vals'])+1)*params['num_agents']

    print('state oh size:', state_oh_size)

    ##Initialize policies & action spaces
    
    if params['scenario'] == 'Basic':
        honest_action_space, honest_action_space_size = getHonestActionSpace(params)
        byz_action_space, byz_action_space_size = getByzantineActionSpace(params)
        
        honest_policy = BasicPolicy(honest_action_space_size, state_oh_size, params['hidden_sizes'], activation, output_activation, params['use_bias']).to(device)
        byz_policy = BasicPolicy(byz_action_space_size, state_oh_size, params['hidden_sizes'], activation, output_activation, params['use_bias']).to(device)

    honest_optimizer = torch.optim.Adam(honest_policy.parameters(), lr=params['learning_rate'])
    byz_optimizer = torch.optim.Adam(byz_policy.parameters(), lr=params['learning_rate'])

    oneHotStateMapper = np.eye(len(params['commit_vals'])+1) # number of unique values that can be in the state. 
    honest_oneHotActionMapper = np.eye(honest_action_space_size)
    byz_oneHotActionMapper = np.eye(byz_action_space_size)
    ## Initialize vpg
    if params['rl_algo_wanted']=='vpg' and params['use_vpg']:
        adv_hidden_sizes = (16,8)
        adv_learning_rate=0.003
        adv_activation= torch.relu
        adv_output_activation = None # I do softmax in the env section.
        adv_use_bias = True
        adv_output_size = 1
        # currently byz and honest use the same network sizes and learning rates.
        honest_v_function = BasicPolicy(adv_output_size, state_oh_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias).to(device)
        honest_q_function = BasicPolicy(adv_output_size, state_oh_size+honest_action_space_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias).to(device)
        honest_v_function_optimizer = torch.optim.Adam(honest_v_function.parameters(), lr=adv_learning_rate)
        honest_q_function_optimizer = torch.optim.Adam(honest_q_function.parameters(), lr=adv_learning_rate)

        byz_v_function = BasicPolicy(adv_output_size, state_oh_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias).to(device)
        byz_q_function = BasicPolicy(adv_output_size, state_oh_size+byz_action_space_size, adv_hidden_sizes, adv_activation, adv_output_activation, adv_use_bias).to(device)
        byz_v_function_optimizer = torch.optim.Adam(byz_v_function.parameters(), lr=adv_learning_rate)
        byz_q_function_optimizer = torch.optim.Adam(byz_q_function.parameters(), lr=adv_learning_rate)

        adv_optimizers = [honest_v_function_optimizer, honest_q_function_optimizer, byz_v_function_optimizer, byz_q_function_optimizer]

        for net in [honest_v_function, honest_q_function, byz_v_function, byz_q_function]:
            net.train()
            net.zero_grad()
        
    if params['load_policy_honest'] != "None":
        print("LOADING IN an honest policy, load_policy=True")
        honest_policy = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_honest']+'.torch')
        if params['use_vpg']:
            honest_v_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_honest']+'_v'+'.torch')
            honest_q_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_honest']+'_q'+'.torch')
    if params['load_policy_byz'] != "None":
        byz_policy = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_byz']+'.torch')
        if params['use_vpg']: 
            byz_v_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_byz']+'_v'+'.torch')
            byz_q_function = torch.load(params['LOAD_PATH_EXPERIMENT']+params['load_policy_byz']+'_q'+'.torch')
            #encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss, curr_ep, best_eval_acc = loadpolicy(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, load_name)

    if params['train_honest']:
        honest_policy.train()
    else: 
        honest_policy.eval()
    if params['train_byz']:
        byz_policy.train() # the value functions i iterate through in main for train().
    else: 
        byz_policy.eval() 

    ################## Finished Initialization ###############
    print('onehotmapper is', oneHotStateMapper)
    print('this script is running first, numb of agents is: ', params['num_agents'])

    honest_policy.zero_grad()
    byz_policy.zero_grad()

    if params['rl_algo_wanted']=='vpg' and params['use_vpg']:
        for net in [honest_v_function, honest_q_function, byz_v_function, byz_q_function]:
            net.train()

    curr_ep = params['starting_ep']
    honest_curr_temperature=params['honest_starting_temp']
    byz_curr_temperature = params['byz_starting_temp']
    temperature_tracker = []

    honest_wins_total = []
    honest_adv_loss_v = []
    honest_adv_loss_q = []
    honest_losses = []
    byzantine_losses = []
    honest_rewards = []
    byzantine_rewards = []
    total_trajectory_logs = []

    total_epochs = params['epochs']*params['byz_honest_train_ratio'] #*params['vpg_epochs_ratio']

    while curr_ep < (total_epochs+params['starting_ep']):  
        print('Epoch', curr_ep)

        honest_optimizer.zero_grad()
        byz_optimizer.zero_grad()

        # vpg only training loop:
        if params['use_vpg']:
            for vpg_ind in range(params['vpg_epochs_ratio']):
                print('Running a VPG training loop')

                for opt in [honest_v_function_optimizer, honest_q_function_optimizer, byz_v_function_optimizer, byz_q_function_optimizer]:
                    opt.zero_grad()

                honest_policy.eval()
                byz_policy.eval()
                
                curr_ep_trajectory_logs, satisfied_constraints, epoch_honest_reward, epoch_byz_reward, hit_max_round_len, avg_round_len = run_iters(params, honest_curr_temperature, byz_curr_temperature, honest_policy, byz_policy, oneHotStateMapper, device)

                losses, adv_losses = rl_algo(curr_ep_trajectory_logs, toOneHotState, 
                toOneHotActions, device,oneHotStateMapper, byz_oneHotActionMapper, 
                honest_oneHotActionMapper, params['send_all_first_round_reward'],
                params['additional_round_penalty'], adv_honest_nets = [honest_v_function, honest_q_function],
                adv_byz_nets = [byz_v_function, byz_q_function], use_vpg=params['use_vpg'])#, honest_action_to_ind, byz_action_to_ind)
                #print("adv losses, should be flattened", adv_losses)

                for ind, adv_loss in enumerate(adv_losses):
                    #print('this adv loss is:', adv_loss)
                    adv_loss.backward()
                    adv_optimizers[ind].step()

        if curr_ep % params['byz_honest_train_ratio']!=0:
            # freeze the weights of the honest
            honest_policy.eval()
            byz_policy.train()
            byz_train_only=True
            #print('training only byz')
        else: 
            honest_policy.train()
            byz_policy.train()
            byz_train_only=False
            #print('training both')

            #only anneal the temperature if we are training both honest and byz. 
            honest_curr_temperature, byz_curr_temperature = temp_annealer(params, honest_curr_temperature, byz_curr_temperature)
            temperature_tracker.append(honest_curr_temperature)

        # clear the vpg gradients if they are being used
        if params['use_vpg']:
            for opt in [honest_v_function_optimizer, honest_q_function_optimizer, byz_v_function_optimizer, byz_q_function_optimizer]:
                opt.zero_grad()

        # Running simultations of the current policy to evaluate it
        curr_ep_trajectory_logs, satisfied_constraints, epoch_honest_reward, epoch_byz_reward, hit_max_round_len, avg_round_len = run_iters(params, honest_curr_temperature, byz_curr_temperature, honest_policy, byz_policy, oneHotStateMapper, device)

        if params['use_vpg']:
            losses, adv_losses = rl_algo(curr_ep_trajectory_logs, toOneHotState, 
            toOneHotActions, device,oneHotStateMapper, byz_oneHotActionMapper, 
            honest_oneHotActionMapper, params['send_all_first_round_reward'],
            params['additional_round_penalty'], adv_honest_nets = [honest_v_function, honest_q_function],
            adv_byz_nets = [byz_v_function, byz_q_function], use_vpg=params['use_vpg'])#, honest_action_to_ind, byz_action_to_ind)
            #print("adv losses, should be flattened", adv_losses)
        else: 
            losses = rl_algo(curr_ep_trajectory_logs, toOneHotState, 
            toOneHotActions, device, oneHotStateMapper, byz_oneHotActionMapper, 
            honest_oneHotActionMapper, params['send_all_first_round_reward'],
            params['additional_round_penalty'], use_vpg=params['use_vpg'])
            #print(losses)
            #print('the loss', losses[0])

        honest_loss = losses[0] # store like this so they are interpretable and can print later if want to. 

        if params['train_honest'] and byz_train_only==False:
            honest_loss.backward()
            honest_optimizer.step()

        honest_losses.append(honest_loss.detach())
        honest_rewards.append(epoch_honest_reward)

        if params['num_byzantine']!=0:
            byz_loss = losses[1]
            if params['train_byz']:
                byz_loss.backward()
                byz_optimizer.step()
            byzantine_losses.append(byz_loss.detach())
            byzantine_rewards.append(epoch_byz_reward)

        # update the advantage functions: 
        if params['use_vpg']:
            #print('adv losses', adv_losses)
            for ind, adv_loss in enumerate(adv_losses):
                #print('this adv loss is:', adv_loss)
                adv_loss.backward()
                adv_optimizers[ind].step()
            
        if params['use_vpg']:
            honest_adv_loss_v.append(adv_losses[0].detach())
            honest_adv_loss_q.append(adv_losses[1].detach())

        #compute and store the losses and other metrics: 
        honest_victory = [ 1 if s==True else 0 for s in satisfied_constraints  ]

        #honest_wins_total += honest_victory
        #byz_rewards = sum([ s[1] for s in satisfied_constraints ])
        honest_wins_total.append(sum(honest_victory)/iters_per_epoch)
        
        # get all of the relevant metrics. eg. loss.item()

        if (curr_ep % params['print_every'] == 0):
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
            print('Honest losses', honest_loss)
            if params['num_byzantine']!=0:
                print( 'Byz losses', byz_loss)
            #print( 'cum sum of honest wins', sum(honest_wins_total)*iters_per_epoch, '=============')
            #print('as a percentage of all trajectories:', (sum(honest_wins_total)*iters_per_epoch)/ (curr_ep*iters_per_epoch))
            print('Current Epoch is: ', curr_ep)
            print('Current Honest Temperature is:' , honest_curr_temperature, 'Byz Temp', byz_curr_temperature, '=======')
            if params['use_vpg']:
                print('Advantage Losses', adv_losses)
            print('Losses from the Epoch', losses)
            print('=============================')
            print('=============================')

        curr_ep += 1

    # plot the change in temperature over time. 
    # plot average honest dddwin rate over time. 
    save_labels = ['honest_wins', 'temperature', 'honest_loss', 'byz_loss', 
    'honest_adv_loss_v', 'honest_adv_loss_q', 'honest_rewards', 'byz_rewards']
    for to_plot, label in zip([honest_wins_total, temperature_tracker, 
    honest_losses, byzantine_losses, honest_adv_loss_v, 
    honest_adv_loss_q, honest_rewards, byzantine_rewards],save_labels):
        savePlot(params, to_plot, label, experiment_name)

    pickle.dump(total_trajectory_logs, open('runs/trajectory_logs-'+experiment_name+'.pickle', 'wb'))

    save_names = ['honest_policy', 'byz_policy']
    save_models = [honest_policy, byz_policy]
    if params['use_vpg']:
        save_names += ['honest_policy_v', 'honest_policy_q', 
         'byz_policy_v', 'byz_policy_q']
        save_models += [honest_v_function, honest_q_function, 
        byz_v_function, byz_q_function]

    for m, n in zip(save_models, save_names):
        torch.save(m, 'saved_models/'+experiment_name+n+'.torch')


# if the policy is better then save it. is overfitting a problem in RL?
def getActivation(activation_string):
    if activation_string is 'tanh':
        return torch.tanh
    if activation_string is 'relu':
        return torch.relu
    if activation_string is 'sigmoid':
        return torch.sigmoid
    if activation_string is 'hardtanh':
        return torch.hardtanh
    if activation_string is 'leakyrelu':
        return torch.leakyrelu
    else:
        return None

def getHonestActionSpace(params):
    if params['scenario']=='Basic':
        honest_action_space = getActionSpace(params, False, byzantine_inds=None, can_send_either_value=params['honest_can_send_either_value'])
        honest_action_space_size = len(honest_action_space)
        #honest_action_to_ind = {a:ind for ind, a in enumerate(honest_action_space)}
    return honest_action_space, honest_action_space_size#, honest_action_to_ind

def getByzantineActionSpace(params):
    if params['scenario']=='Basic':  
        byz_action_space = getActionSpace(params, True, byzantine_inds=[0], can_send_either_value=params['honest_can_send_either_value'])
        byz_action_space_size = len(byz_action_space)
        #byz_action_to_ind = {a:ind for ind, a in enumerate(byz_action_space)}
        #print('byz action to ind', )
    return byz_action_space, byz_action_space_size#, byz_action_to_ind



if __name__=='__main__':
    main()