''' Configuration file storing all of the intialization details '''

import datetime
import torch 
from nn import *

# Experiment Settings
experiment_base_name = 'Basic_Test_Run'
random_seed = 27

load_policy = False
load_name = 'Lolz'

print_every = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Environment Settings
scenario = 'Basic'
#environment = getEnv(scenario)
#if scenario=='Basic':
commit_vals = (0,1)
null_message_val = 2
num_agents = 3
num_byzantine = 1

oneHotMapper = dict()
for com_val in commit_vals:
    z = np.zeros(len(commit_vals))
    z[com_val] = 1
    oneHotMapper[com_val]=z

oneHotMapper[null_message_val]=np.zeros(len(commit_vals))

print('this script is running first, numb of agents is: ', num_agents)

# Training Settings
epochs = 50
iters_per_epoch = 10
max_ep_len=1000 # max number of rounds before termination of the current simulation

# NN Settings
learning_rate=0.001
batch_size = 32

if load_policy:
    print("LOADING IN A policy, load_policy=True")
    #encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss, curr_ep, best_eval_acc = loadpolicy(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, load_name)

else: 
    if scenario=='Basic':
        #honest_policy = BasicPolicyHonest(device).to(device)
        #byz_policy = BasicPolicyByz(device).to(device)
        pass

    # cant be 0 else later on there is division by zero!
    curr_ep = 1 

#policy.train()
#optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

mem_pin = False


# lip=15 if want this see Protein AE code to add it. 

# RL Settings
rl_algo = 'ppo'
steps_per_epoch=4000
gamma=0.99
clip_ratio=0.2
vf_lr=1e-3
train_policy_iters=80
train_vf_iters=80
lam=0.97
target_kl=0.01
logger_kwargs=dict()
save_freq=10

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
epochs, iters_per_epoch, rl_algo, date_time )