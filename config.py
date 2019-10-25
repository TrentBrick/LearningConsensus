''' Configuration file storing all of the intialization details '''

import datetime
import torch 
from nn import *
from rl_algo import vpg


import itertools
from collections import OrderedDict
def getActionSpace(isByzantine, byzantine_inds=None, can_send_either_value=True):
    '''move this to a new script that config and environment and agent utils can reference. '''

    # making the action space. 
    # this depends on if the agent is Byzantine or not. 
    ## if byzantine then global view over all Byzantines, 
    # means that the same agent gets to act multiple times in the round
    # for honest, they only currently have option to send to everyone. 
    parties = set(range(num_agents))
    if scenario == 'Basic':

        action_space = []

        if isByzantine:
            action_space.append('no_send')
            # no point in sending messages to other Byzantines as the central agent knows what the states are
            # but do have v granular send options. 
            # and no commit option
            # get every possible combination of sending actions possible
            
            # remove the honest agents. 
            non_byzantines = list(range(0, num_agents))
            for byzantine_ind in byzantine_inds:
                non_byzantines.remove(byzantine_ind)
            #for val in commit_vals:
            #    non_byzantines.append('v'+str(val)) # add in the possible values that can be sent

            # this code is tricky, I get all combinations of the honest agents to send to
            # and then interleave in all permutations of the values that can be sent to them. 
            # for example a subset of them is: : 
            ''''send_agent-2_value-0',
            'send_agent-1_value-0_agent-3_value-0',
            'send_agent-1_value-0_agent-2_value-1_agent-3_value-1',
            'send_agent-1_value-1_agent-3_value-1',
            'send_agent-1_value-1_agent-2_value-1_agent-3_value-0',
            'send_agent-1_value-0_agent-2_value-0_agent-3_value-0',
            'send_agent-1_value-1_agent-2_value-1',
            'send_agent-2_value-1_agent-3_value-1','''
            for choose_n in range(1, len(non_byzantines)+1):
                commit_val_permutes = list(itertools.permutations(commit_vals*((choose_n//2)+1)))
                for combo_el in itertools.combinations(non_byzantines, choose_n):
                    for cvp in commit_val_permutes:
                        string = 'send'
                        for ind in range(choose_n):
                            string += '_agent-'+str(combo_el[ind])+'_v-'+str(cvp[ind])
                        action_space.append( string ) 
            # remove any redundancies in a way that preserves order. 
            action_space = list(OrderedDict.fromkeys(action_space))

        else:
            if can_send_either_value: 
                for commit_val in commit_vals:
                    action_space.append('send_to_all-value_'+str(commit_val)) 
                    action_space.append('commit_'+str(commit_val)) 
            else: 
                action_space.append('send_to_all-value_init') 
                for commit_val in commit_vals:
                    action_space.append('commit_'+str(commit_val)) 

    return action_space

# Experiment Settings
experiment_base_name = 'Adv_Func_Test_OnlyHonest'
directory = 'runs/'
random_seed = 27

load_policy = False
load_name = 'Lolz'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Environment Settings
scenario = 'Basic'
#environment = getEnv(scenario)
#if scenario=='Basic':
commit_vals = (0,1)
# assumes honest and byz see current state and only current state for now. 
# own state as a len 2 vector * num agents 

num_agents = 3
num_byzantine = 0 #currently will not work for any larger values than 1!!!! 

# Training Settings
epochs = 500
iters_per_epoch = 200 # I think this number is really important to tune. 
max_round_len=1000 # max number of rounds before termination of the current simulation
print_every = 5

# RL Settings
starting_temp = 6 # this is so high to try and encourage lots of exploration
temp_anneal = 0.985 #5 is a bit better 0.99 before. 
temp_fix_point = 1.0
honest_can_send_either_value = False # can the honest agents send only their init value or other values also? 
use_heat_jumps = False # when it hits the temp fix point, increase the temp back to the starting temp. 
rl_algo_wanted = 'vpg'
if rl_algo_wanted=='vpg':
    rl_algo = vpg

# lots of these refer to PPO which will be implemented later. 
steps_per_epoch=4000
gamma=0.999
clip_ratio=0.2
vf_lr=1e-3
train_policy_iters=80
train_vf_iters=80
lam=0.97
target_kl=0.01
logger_kwargs=dict()
save_freq=10

# penalties/rewards. (honest ,byzantine) 
# ensure that the commit first hurts more than even getting it wrong. 
dishonesty_violation = np.array([0,0]) # doesnt send its value to everyone else on the first round. 
consistency_violation = np.array([-1, 1])
validity_violation = np.array([-2, 1]) # used to be -0.75
majority_violation = np.array([-1, 1]) # -0.5
correct_commit = np.array([1, -1])
round_penalty = np.array([0,0]) # currently only applies to the honest parties

# NN Settings
learning_rate=0.003
batch_size = 32
hidden_sizes = (16,8)
activation= torch.tanh
output_activation = None # I do softmax in the env section. 
use_bias = True
starting_ep = 1 # dont change
# size of the vector input to the neural network. this is the states that all other agents have. 
state_oh_size = (len(commit_vals)+1)*num_agents
null_message_val = 2

if load_policy:
    print("LOADING IN A policy, load_policy=True")
    #encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss, curr_ep, best_eval_acc = loadpolicy(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, load_name)

else: 
    if scenario=='Basic':
        honest_action_space = getActionSpace(False, byzantine_inds=None, can_send_either_value=honest_can_send_either_value)
        honest_action_space_size = len(honest_action_space)
        honest_action_to_ind = {a:ind for ind, a in enumerate(honest_action_space)}
        
        byz_action_space = getActionSpace(True, byzantine_inds=[0], can_send_either_value=honest_can_send_either_value)
        byz_action_space_size = len(byz_action_space)
        byz_action_to_ind = {a:ind for ind, a in enumerate(byz_action_space)}

        honest_policy = BasicPolicy(honest_action_space_size, state_oh_size, hidden_sizes, activation, output_activation, use_bias).to(device)
        byz_policy = BasicPolicy(byz_action_space_size, state_oh_size, hidden_sizes, activation, output_activation, use_bias).to(device)

    honest_optimizer = torch.optim.Adam(honest_policy.parameters(), lr=learning_rate)
    byz_optimizer = torch.optim.Adam(byz_policy.parameters(), lr=learning_rate)
    # cant be 0 else later on there is division by zero!

honest_policy.train()
byz_policy.train()

oneHotStateMapper = np.eye(len(commit_vals)+1)
honest_oneHotActionMapper = np.eye(honest_action_space_size)
byz_oneHotActionMapper = np.eye(byz_action_space_size)
'''for com_val in commit_vals:
    z = np.zeros(len(commit_vals)+1)
    z[com_val] = 1
    oneHotMapper[com_val]=z

null = np.zeros(len(commit_vals)+1)
null[-1] = 1
oneHotMapper[null_message_val]=null'''
print('onehotmapper is', oneHotStateMapper)
print('this script is running first, numb of agents is: ', num_agents)

##################
# setting up the advantage function estimators
# first the value function. input is the state and output is a real number. 
if rl_algo_wanted=='vpg':
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

    for net in [honest_v_function, honest_q_function, byz_v_function, byz_q_function]:
        net.train()

adv_optimizers = [honest_v_function_optimizer, honest_q_function_optimizer, byz_v_function_optimizer, byz_q_function_optimizer]

mem_pin = False # if you want to put your data in the gpu. we dont have data here so not sure if this would do anything... 
# clip=15 if want this see my (Trenton's) Protein AE code to add it. 

'''
if load_policy:
    print("LOADING IN A policy, load_policy=True")
    #encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss, curr_ep, best_eval_acc = loadpolicy(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, load_name)

else: 
    if rl_algo=='ppo':
        vf_model = VFModel(device).to(device)

vf_optimizer = torch.optim.Adam(vf_model.parameters(), lr=vf_lr)
'''

    # init the necessary/expected policys here. 

date_time = str(datetime.datetime.now()).replace(' ', '_')
# used to save the policy and its outputs. 
experiment_name = experiment_base_name+"rand_seed-%s_scenario-%s_epochs-%s_iters_per_ep-%s_rl_algo-%s_time-%s" % (random_seed, scenario, 
epochs, iters_per_epoch, str(rl_algo), date_time )