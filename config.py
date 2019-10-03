''' Configuration file storing all of the intialization details '''

import datetime
import torch 
from nn import *
from rl_algo import *


import itertools
from collections import OrderedDict
def getActionSpace(isByzantine, byzantine_inds=None):
    '''move this to a new script that config and environment and agent utils can reference. '''

    # making the action space. 
    # this depends on if the agent is Byzantine or not. 
    ## if byzantine then global view over all Byzantines, 
    # means that the same agent gets to act multiple times in the round
    # for honest, they only currently have option to send to everyone. 
    parties = set(range(num_agents))
    if scenario == 'Basic':

        action_space = []#['no_send']

        if isByzantine:
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
            for commit_val in commit_vals:
                action_space.append('send_to_all-value_'+str(commit_val)) 
                action_space.append('commit_'+str(commit_val)) 

    return action_space



# Experiment Settings
experiment_base_name = 'Basic_Test_Run_OnlyHonest'
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

state_oh_size = len(commit_vals)*num_agents # commit values one hot len * commit value+no message
null_message_val = 2

oneHotMapper = dict()
for com_val in commit_vals:
    z = np.zeros(len(commit_vals))
    z[com_val] = 1
    oneHotMapper[com_val]=z

oneHotMapper[null_message_val]=np.zeros(len(commit_vals))

print('this script is running first, numb of agents is: ', num_agents)

# Training Settings
epochs = 5000
iters_per_epoch = 50
max_round_len=10 # max number of rounds before termination of the current simulation
print_every = 1

# NN Settings
learning_rate=0.00001
batch_size = 32
hidden_sizes = (32,32,)
activation= torch.tanh
output_activation = None # I do softmax in the env section. 
use_bias = True

if load_policy:
    print("LOADING IN A policy, load_policy=True")
    #encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss, curr_ep, best_eval_acc = loadpolicy(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, load_name)

else: 
    if scenario=='Basic':

        honest_action_space_size = len(getActionSpace(False, byzantine_inds=None))
        byz_action_space_size = len(getActionSpace(True, byzantine_inds=[0]))

        honest_policy = BasicPolicy(honest_action_space_size, state_oh_size, hidden_sizes, activation, output_activation, use_bias).to(device)
        byz_policy = BasicPolicy(byz_action_space_size, state_oh_size, hidden_sizes, activation, output_activation, use_bias).to(device)

    honest_optimizer = torch.optim.Adam(honest_policy.parameters(), lr=learning_rate)
    byz_optimizer = torch.optim.Adam(byz_policy.parameters(), lr=learning_rate)
    # cant be 0 else later on there is division by zero!
starting_ep = 1 

honest_policy.train()
byz_policy.train()

mem_pin = False
# clip=15 if want this see Protein AE code to add it. 

# RL Settings
starting_temp = 300 # this is so high to try and encourage lots of exploration
temp_anneal = 0.999
temp_fix_point = 1.0
use_heat_jumps = False # when it hits the temp fix point, increase the temp back to the starting temp. 
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
commit_first_round_penalty = np.array([-0,0])
consistency_violation = np.array([-1, 1])
validity_violation = np.array([-0.75, 1])
majority_violation = np.array([-0.5, 1])
correct_commit = np.array([1, -1])
round_penalty = np.array([-0.01,0.1]) # currently only applies to the honest parties

# need to make the RL neural networks: 
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